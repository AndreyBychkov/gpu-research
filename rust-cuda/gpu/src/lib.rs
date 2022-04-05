#![cfg_attr(
target_os = "cuda",
no_std,
feature(register_attr),
register_attr(nvvm_internal)
)]

#![allow(improper_ctypes_definitions)]

use cuda_std::prelude::*;
use cuda_std::shared_array;
use cuda_std::thread::{block_dim_x, block_dim_y, block_idx_x, block_idx_y, sync_threads, thread_idx_x, thread_idx_y};

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

const TILE_SZ: usize = 32;
type TileArray = [f32; TILE_SZ];

#[kernel]
pub unsafe fn matmul_tiled(a: &[f32], b: &[f32], out: *mut f32, n: usize) {
    let row = (block_idx_y() * block_dim_y() + thread_idx_y()) as usize;
    let col = (block_idx_x() * block_dim_x() + thread_idx_x()) as usize;
    let thr_idx_y = thread_idx_y() as usize;
    let thr_idx_x = thread_idx_x() as usize;

    let mut tile_a = shared_array![f32; TILE_SZ * TILE_SZ];
    let mut tile_b = shared_array![f32; TILE_SZ * TILE_SZ];

    let mut sum = 0f32;
    let block_n = (TILE_SZ + n - 1) / TILE_SZ;
    for i in 0..block_n {
        *tile_a.add(thr_idx_y * TILE_SZ + thr_idx_x) =
            if TILE_SZ * i + thr_idx_x < n && row < n {
                a[row * n + i * TILE_SZ + thr_idx_x]
            } else {
                0f32
            };

        *tile_b.add(thr_idx_y * TILE_SZ + thr_idx_x) =
            if TILE_SZ * i + thr_idx_y < n && col < n {
                b[n * (i * TILE_SZ + thr_idx_y) + col]
            } else {
                0f32
            };

        sync_threads();
        for j in 0..TILE_SZ {
            sum += *tile_a.add(thr_idx_y * TILE_SZ + j) * (*tile_b.add(j * TILE_SZ + thr_idx_x));
        }
        sync_threads();
    }
    if row < n && col < n {
        let out_place = &mut *out.add(row * n + col);
        *out_place = sum;
    }
}



