// Builds csrc/llama_shim.c into a static library with the system C compiler and
// links it against the *prebuilt* libllama.so (no cc crate: offline build).
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=csrc/llama_shim.c");
    println!("cargo:rerun-if-env-changed=LLAMA_DIR");
    if env::var_os("CARGO_FEATURE_LLAMA").is_none() {
        return;
    }
    let llama = PathBuf::from(env::var("LLAMA_DIR").unwrap_or_else(|_| "/workspace/llama.cpp".into()));
    let lib_dir = llama.join("build/bin");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj = out.join("llama_shim.o");
    let cc = env::var("CC").unwrap_or_else(|_| "cc".into());
    let st = Command::new(&cc)
        .args(["-O2", "-std=c11", "-fPIC", "-c", "csrc/llama_shim.c", "-o"])
        .arg(&obj)
        .arg(format!("-I{}", llama.join("include").display()))
        .arg(format!("-I{}", llama.join("ggml/include").display()))
        .status()
        .expect("failed to run C compiler");
    assert!(st.success(), "compiling csrc/llama_shim.c failed");
    let lib = out.join("libllama_shim.a");
    let _ = std::fs::remove_file(&lib);
    let st = Command::new("ar").arg("crs").arg(&lib).arg(&obj).status().expect("ar failed");
    assert!(st.success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=llama_shim");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}
