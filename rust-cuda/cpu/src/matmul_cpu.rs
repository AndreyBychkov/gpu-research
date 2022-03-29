use rayon::prelude::*;
use rayon::iter::{ParallelIterator, IntoParallelRefIterator};

pub fn matmul_cpu(l: &Vec<f32>, r: &Vec<f32>) -> Vec<f32> {
    let mut res = vec![0.0; l.len()];
    let n = (l.len() as f32).sqrt().round() as usize;

    let step_row = |(i, r_row): (usize, &mut [f32])| {
        for (j, res_row) in r_row.iter_mut().enumerate() {
            let mut c = 0.0;
            for k in 0..n {
                let x = l[n * i + k];
                let y = r[n * k + j];
                c += x * y;
            }
            *res_row = c;
        }
    };

    res.par_chunks_mut(n)
        .enumerate()
        .for_each(step_row);

    res
}