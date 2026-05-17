fn main() {
    // NET_FILE_SIZE = (INPUT_SIZE × HIDDEN_SIZE × 2) + (HIDDEN_SIZE × 4) + 2
    //              = (704 × 32 × 768 × 2) + (768 × 4) + 2 = 34,606,082
    // l0w: INPUT_SIZE×HIDDEN_SIZE i16 (×2), l0b: HIDDEN_SIZE i16 (×2),
    // l1w: HIDDEN_SIZE×2 i8 (×1), l1b: 1 i16 (×2)
    let expected: usize = 704 * 32 * 768 * 2 + 768 * 4 + 2;
    let src = std::path::Path::new("nets/default.nnue");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("default.nnue");

    if src.metadata().map(|m| m.len() as usize >= expected).unwrap_or(false) {
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
