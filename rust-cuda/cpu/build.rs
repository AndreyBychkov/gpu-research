mod src;

use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../gpu")
        .copy_to("../resources/kernels.ptx")
        .build()
        .unwrap();
}