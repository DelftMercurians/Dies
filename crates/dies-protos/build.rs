extern crate glob;
extern crate protobuf_codegen;

use protobuf_codegen::Codegen;

fn main() {
    // Find all .proto files in src/protos
    let proto_files = glob::glob("src/protos/*.proto")
        .expect("Failed to read glob pattern")
        .map(|entry| entry.unwrap().to_owned())
        .collect::<Vec<_>>();

    Codegen::new()
        .pure()
        .cargo_out_dir("protos")
        .inputs(&proto_files)
        .include("src/protos")
        .run_from_script();
}
