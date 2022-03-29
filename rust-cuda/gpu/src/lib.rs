#![cfg_attr(
target_os = "cuda",
no_std,
feature(register_attr),
register_attr(nvvm_internal)
)]

#![allow(improper_ctypes_definitions)]

use cuda_std::prelude::*;
use cuda_std::thread::{block_dim_x, block_dim_y, block_idx, block_idx_x, block_idx_y, thread_idx_x, thread_idx_y};

#[kernel]
pub unsafe fn add(a: &[f32], b: &[f32], c: *mut f32) {
    let idx = thread::index_1d() as usize;
    if idx < a.len() {
        let elem = &mut *c.add(idx);
        *elem = a[idx] + b[idx];
    }
}

#[kernel]
pub unsafe fn matmul(a: &[f32], b: &[f32], out: *mut f32, n: usize) {
    let row = (block_idx_y() * block_dim_y() + thread_idx_y()) as usize;
    let col = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
    if row < n && col < n {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += a[row * n + i] * b[i * n + col];
        }
        let out_place = &mut *out.add(row * n + col);
        *out_place = sum;
    }
}

