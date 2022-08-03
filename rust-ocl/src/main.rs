extern crate ocl;
use ocl::{ProQue};

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
const N: usize = 64;
const N2: usize = N*N;

fn main() -> ocl::Result<()> {
    let pro_que = ProQue::builder()
        .src(PROGRAM_SOURCE)
        .dims(N2)
        .build()?;

    let buffer_a = pro_que.create_buffer::<f32>()?;
    let buffer_b = pro_que.create_buffer::<f32>()?;
    let buffer_c = pro_que.create_buffer::<f32>()?;

    let mut vec_a = vec![1.0f32; N2];
    let mut vec_b = vec![1.0f32; N2];

    buffer_a.write(&vec_a).enq()?;
    buffer_b.write(&vec_b).enq()?;
    pro_que.flush()?;

    let kernel = pro_que.kernel_builder(KERNEL_NAME)
        .global_work_size(N)
        .arg(&buffer_a)
        .arg(&buffer_b)
        .arg(&buffer_c)
        .arg(64)
        .build()?;

    unsafe { kernel.enq()?; }
    let mut vec_c = vec![1.0f32; buffer_c.len()];
    buffer_c.read(&mut vec_c).enq()?;

    for row in vec_c.chunks(N) {
        println!("{:?} ", row);
    }

    Ok(())

}
