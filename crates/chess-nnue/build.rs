fn main() {
    // Minimum accepted size = legacy single-output-bucket format:
    // (INPUT_SIZE × HIDDEN_SIZE × 2) + (HIDDEN_SIZE × 4) + 2 = 37,751,810.
    // The 8-output-bucket format is larger (extra l1w rows + biases) and also
    // passes this check; NnueNetwork::from_bytes dispatches on actual size.
    let expected: usize = 768 * 32 * 768 * 2 + 768 * 4 + 2;
    let src = std::path::Path::new("nets/default.nnue");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("default.nnue");

    if src
        .metadata()
        .map(|m| m.len() as usize >= expected)
        .unwrap_or(false)
    {
        std::fs::copy(src, &out).unwrap();
    } else {
        std::fs::write(&out, vec![0u8; expected]).unwrap();
        println!(
            "cargo:warning=nets/default.nnue has wrong size ({} bytes, expected {expected}) — falling back to HCE until retrained",
            src.metadata().map(|m| m.len()).unwrap_or(0)
        );
    }
    println!("cargo:rerun-if-changed=nets/default.nnue");
}
