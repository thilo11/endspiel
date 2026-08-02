fn main() {
    const MAGIC: &[u8; 8] = b"ESPNNUE2";
    const INPUT_SIZE: usize = 768 * 32;
    const FT_SIZE: usize = 1024;
    const OUTPUT_BUCKETS: usize = 8;
    const L1_SIZE: usize = 16;
    const L2_SIZE: usize = 32;
    const HEADER_SIZE: usize = 32;
    const EXPECTED: usize = HEADER_SIZE
        + INPUT_SIZE * FT_SIZE * 2
        + FT_SIZE * 2
        + OUTPUT_BUCKETS * L1_SIZE * FT_SIZE
        + OUTPUT_BUCKETS * L1_SIZE * 4
        + OUTPUT_BUCKETS * L2_SIZE * L1_SIZE
        + OUTPUT_BUCKETS * L2_SIZE * 4
        + OUTPUT_BUCKETS * L2_SIZE
        + OUTPUT_BUCKETS * 4;

    let src = std::path::Path::new("nets/default.nnue");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("default.nnue");

    if src
        .metadata()
        .map(|metadata| (EXPECTED..EXPECTED + 64).contains(&(metadata.len() as usize)))
        .unwrap_or(false)
    {
        std::fs::copy(src, &out).unwrap();
    } else {
        let mut placeholder = vec![0u8; EXPECTED];
        placeholder[..8].copy_from_slice(MAGIC);
        for (idx, value) in [
            1u32,
            INPUT_SIZE as u32,
            FT_SIZE as u32,
            OUTPUT_BUCKETS as u32,
            L1_SIZE as u32,
            L2_SIZE as u32,
        ]
        .into_iter()
        .enumerate()
        {
            let start = 8 + idx * 4;
            placeholder[start..start + 4].copy_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&out, placeholder).unwrap();
        println!(
            "cargo:warning=nets/default.nnue is not a layer-stacked net — using an untrained placeholder"
        );
    }
    println!("cargo:rerun-if-changed=nets/default.nnue");
}
