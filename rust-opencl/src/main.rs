use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_event, cl_float, CL_BLOCKING, CL_NON_BLOCKING};
use opencl3::Result;
use std::ptr;
use nanorand::{Rng, WyRand};


const PROGRAM_SOURCE: &str = r#"
__kernel void matmul(__global float* A,
                     __global float* B,
                     __global float* C,
                     int N) {
    int i = get_global_id(0);
    for (size_t j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (size_t k = 0; k < N; ++k) {
            acc += A[i*N + k] * B[k*N + j];
        }
        C[i*N + j] = acc;
    }
}"#;

const KERNEL_NAME: &str = "matmul";

fn main() -> Result<()> {
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)?
        .first()
        .expect("no device found in platform");
    let device = Device::new(device_id);

    let context = Context::from_device(&device).expect("Context::from_device failed");

    // Create a command_queue on the Context's device
    let queue = CommandQueue::create(
        &context,
        context.default_device(),
        CL_QUEUE_PROFILING_ENABLE,
    ).expect("CommandQueue::create failed");

    // Build the OpenCL program source and create the kernel.
    let program = Program::create_and_build_from_source(&context, PROGRAM_SOURCE, "")
        .expect("Program::create_and_build_from_source failed");
    let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

    const N: usize = 64*64;
    let mut wyrand = WyRand::new();
    let mut lhs = vec![0.0f32; N];
    wyrand.fill(&mut lhs);
    let mut rhs = vec![0.0f32; N];
    wyrand.fill(&mut rhs);

    let mut lhs_gpu = Buffer::<cl_float>::create(&context,
                                                 CL_MEM_READ_ONLY,
                                                 N,
                                                 ptr::null_mut())?;
    let mut rhs_gpu = Buffer::<cl_float>::create(&context,
                                                 CL_MEM_READ_ONLY,
                                                 N,
                                                 ptr::null_mut())?;
    let out_gpu = Buffer::<cl_float>::create(&context,
                                             CL_MEM_WRITE_ONLY,
                                             N,
                                             ptr::null_mut())?;

    // Blocking write
    let lhs_write_event = queue.enqueue_write_buffer(&mut lhs_gpu, CL_NON_BLOCKING, 0, &lhs, &[])?;
    let rhs_write_event = queue.enqueue_write_buffer(&mut rhs_gpu, CL_NON_BLOCKING, 0, &rhs, &[])?;
    // let n = (N as f32).sqrt().round() as i32;
    let n = 64;
    // Use the ExecuteKernel builder to set the kernel buffer and
    // cl_float value arguments, before setting the one dimensional
    // global_work_size for the call to enqueue_nd_range.
    // Unwraps the Result to get the kernel execution event.
    let kernel_event = ExecuteKernel::new(&kernel)
        .set_arg(&lhs_gpu)
        .set_arg(&rhs_gpu)
        .set_arg(&out_gpu)
        .set_arg(&n)
        .set_global_work_size(N)
        // .set_local_work_size(128)
        .set_wait_event(&lhs_write_event)
        .set_wait_event(&rhs_write_event)
        .enqueue_nd_range(&queue)
        .expect("Kernel failed");

    // println!("{}, {}", opencl3::device::CL_DEVICE_MAX_GLOBAL_VARIABLE_SIZE, opencl3::device::CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);

    let mut events: Vec<cl_event> = Vec::default();
    events.push(kernel_event.get());

    // Create a results array to hold the results from the OpenCL device
    // and enqueue a read command to read the device buffer into the array
    // after the kernel event completes.
    let mut out = vec![0.0f32; N];
    let read_event = queue.enqueue_read_buffer(&out_gpu, CL_NON_BLOCKING, 0, &mut out, &events)?;

    // Wait for the read_event to complete.
    read_event.wait()?;

    // Output the first and last results
    println!("results front: {}", &out[0]);
    println!("results back: {:?}", &out[N-1]);

    // Calculate the kernel duration, from the kernel_event
    let start_time = kernel_event.profiling_command_start()?;
    let end_time = kernel_event.profiling_command_end()?;
    let duration = end_time - start_time;
    println!("kernel execution duration (s): {}", duration as f32 / 10f32.powf(9.0));

    Ok(())
}