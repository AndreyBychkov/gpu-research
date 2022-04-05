mod matmul_cpu;
mod matmul_gpu;

use cust::prelude::*;
use nanorand::{Rng, WyRand};
use std::error::Error;
use crate::matmul_cpu::matmul_cpu;
use crate::matmul_gpu::matmul_gpu;
use approx::*;
use std::time::Instant;

const N: usize = 1024;
const N2: usize = (N * N) as usize;

static PTX: &str = include_str!("../../resources/kernels.ptx");

fn add_task() -> Result<(), Box<dyn Error>> {
    let mut wyrand = WyRand::new();
    let mut lhs = vec![2.0f32; N2];
    wyrand.fill(&mut lhs);
    let mut rhs = vec![0.0f32; N2];
    wyrand.fill(&mut rhs);

    // initialize CUDA, this will pick the first available device and will
    // make a CUDA context from it.
    // We don't need the context for anything but it must be kept alive.
    let _ctx = cust::quick_init()?;

    // Make the CUDA module, modules just house the GPU code for the kernels we created.
    // they can be made from PTX code, cubins, or fatbins.
    let module = Module::from_ptx(PTX, &[])?;

    // make a CUDA stream to issue calls to. You can think of this as an OS thread but for dispatching
    // GPU calls.
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    // allocate the GPU memory needed to house our numbers and copy them over.
    let lhs_gpu = lhs.as_slice().as_dbuf()?;
    let rhs_gpu = rhs.as_slice().as_dbuf()?;

    // allocate our output buffer. You could also use DeviceBuffer::uninitialized() to avoid the
    // cost of the copy, but you need to be careful not to read from the buffer.
    let mut out = vec![0.0f32; N2];
    let out_buf = out.as_slice().as_dbuf()?;

    // retrieve the add kernel from the module so we can calculate the right launch config.
    let func = module.get_function("add")?;

    // use the CUDA occupancy API to find an optimal launch configuration for the grid and block size.
    // This will try to maximize how much of the GPU is used by finding the best launch configuration for the
    // current CUDA device/architecture.
    let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;

    let grid_size = (N2 as u32 + block_size - 1) / block_size;

    println!(
        "using {} blocks and {} threads per block",
        grid_size, block_size
    );

    // Actually launch the GPU kernel. This will queue up the launch on the stream, it will
    // not block the thread until the kernel is finished.
    unsafe {
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            func<<<grid_size, block_size, 0, stream>>>(
                lhs_gpu.as_device_ptr(),
                lhs_gpu.len(),
                rhs_gpu.as_device_ptr(),
                rhs_gpu.len(),
                out_buf.as_device_ptr(),
            )
        )?;
    }

    stream.synchronize()?;

    // copy back the data from the GPU.
    out_buf.copy_to(&mut out)?;

    println!("{} + {} = {}", lhs[0], rhs[0], out[0]);
    Ok(())
}

fn matmul_task() -> Result<(), Box<dyn Error>> {
    let mut wyrand = WyRand::new();
    let mut lhs = vec![2.0f32; N2];
    wyrand.fill(&mut lhs);
    let mut rhs = vec![0.0f32; N2];
    wyrand.fill(&mut rhs);

    // let out_cpu = matmul_cpu(&lhs, &rhs);
    let start = Instant::now();
    let out_gpu = matmul_gpu(&lhs, &rhs).expect("Problem with gpu code");
    let elapsed = start.elapsed();
    println!("{}s elapsed", elapsed.as_secs_f32());
    // for (c, g) in out_cpu.iter().zip(out_gpu.iter()) {
    //     abs_diff_eq!(c, g);
    // }
    println!("Ok!");
    Ok(())
}


fn main() -> Result<(), Box<dyn Error>> {
    matmul_task()
}
