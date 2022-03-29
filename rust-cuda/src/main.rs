use cust::prelude::*;
use nanorand::{Rng, WyRand};
use std::error::Error;

#[path = "../cpu/src/mod.rs"]
mod matmul_cpu;

#[path = "../cpu/src/mod.rs"]
mod matmul_gpu;

use approx::*;
use std::time::Instant;
use crate::matmul_cpu::matmul_cpu::matmul_cpu;
use crate::matmul_cpu::matmul_gpu::matmul_gpu;

const N: usize = 1024;
const N2: usize = (N * N) as usize;

static PTX: &str = include_str!("../resources/kernels.ptx");

fn main() -> Result<(), Box<dyn Error>> {
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
