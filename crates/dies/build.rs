use std::path::PathBuf;

extern crate glob;
extern crate protobuf_codegen;

use protobuf_codegen::Codegen;

fn main() {
    // Find all .proto files in src/protos
    let proto_files = glob::glob("src/protos/*.proto")
        .expect("Failed to read glob pattern")
        .map(|entry| entry.unwrap().to_owned())
        .collect::<Vec<_>>();

    // // Print the proto files
    // println!("Proto files:");
    // for file in &proto_files {
    //     println!("{}", file.display());
    // }

    // Compile the proto files
    // prot::compile_protos(&proto_files, &[PathBuf::from("src/protos")]).unwrap();
    Codegen::new()
        .pure()
        // .customize(protobuf_codegen::Customize::default().)
        .cargo_out_dir("protos")
        .inputs(&proto_files)
        .include("src/protos")
        .run_from_script();

    // // Print generated files
    // println!("Generated files:");
    // let generated_files = glob::glob(&format!("{}/**/*.rs", out_dir))
    //     .expect("Failed to read glob pattern")
    //     .map(|entry| entry.unwrap().to_owned())
    //     .collect::<Vec<_>>();
    // for file in generated_files {
    //     println!("{}", file.display());
    // }
    // panic!("Stop here");

    // // Generate a mod.rs file for the generated code
    // let out_dir = std::env::var("OUT_DIR").unwrap();
    // let dest_path = PathBuf::from(out_dir).join("protos.rs");
    // let content = proto_files
    //     .iter()
    //     .map(|path| {
    //         let name = path.file_stem().unwrap().to_str().unwrap();
    //         format!(
    //             "pub mod {0} {{ include!(concat!(env!(\"OUT_DIR\"), \"/protos/{0}.rs\")); }}",
    //             name
    //         )
    //     })
    //     .collect::<Vec<_>>()
    //     .join("\n");
    // std::fs::write(&dest_path, content).unwrap();
}
