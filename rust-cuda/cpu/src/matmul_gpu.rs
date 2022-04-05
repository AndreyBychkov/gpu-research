use cust::prelude::*;
use nanorand::{Rng, WyRand};
use std::error::Error;
use std::time::Instant;

static PTX: &str = include_str!("../../resources/kernels.ptx");

pub(crate) fn matmul_gpu(lhs: &Vec<f32>, rhs: &Vec<f32>) -> Result<Vec<f32>, Box<dyn Error>> {
    let N2 = lhs.len();
    let N = (N2 as f32).sqrt().round() as usize;

    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let lhs_gpu = lhs.as_slice().as_dbuf()?;
    let rhs_gpu = rhs.as_slice().as_dbuf()?;

    let mut out = vec![0.0f32; N2];
    let out_buf = out.as_slice().as_dbuf()?;

    let func = module.get_function("matmul_tiled")?;
    // let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;
    let block_size = 32;
    let grid_size = (N as u32 + block_size - 1) / block_size;
    let dim_block = (block_size, block_size, 1);
    let dim_grid = (grid_size, grid_size, 1);

    println!(
        "using {} blocks and {} threads per block ({} threads total)",
        grid_size*grid_size, block_size*block_size, grid_size*grid_size*block_size*block_size
    );

    let start = Instant::now();
    unsafe {
        launch!(
            func<<<dim_grid, dim_block, 0, stream>>>(
                lhs_gpu.as_device_ptr(),
                lhs_gpu.len(),
                rhs_gpu.as_device_ptr(),
                rhs_gpu.len(),
                out_buf.as_device_ptr(),
                N
            )
        )?;
    }

    stream.synchronize()?;
    let elapsed = start.elapsed();
    println!("Spent on kernel = {}s", elapsed.as_secs_f32());

    out_buf.copy_to(&mut out)?;
    Ok(out)
}