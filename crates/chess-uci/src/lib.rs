pub mod protocol;
pub mod handler;

/// Run the UCI protocol loop.
pub fn run() {
    handler::UciHandler::new().run();
}

/// Run the UCI protocol loop with Syzygy tablebases pre-loaded from `path`.
pub fn run_with_syzygy(path: &str) {
    let mut handler = handler::UciHandler::new();
    handler.set_syzygy(path);
    handler.run();
}
