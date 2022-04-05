use cuda_builder::CudaBuilder;

fn main() {
    let result = CudaBuilder::new("../gpu")
        .copy_to("../resources/kernels.ptx")
        .release(true)
        .build();
    match result {
        Ok(content) => {println!("All good")}
        Err(error) => { println!("error: {}", error); panic!(); }
    }
}