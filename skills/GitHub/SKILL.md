# GitHub & Rust Project Skill

## Commit Messages
Follow Conventional Commits: `type(scope): description`
- Types: feat, fix, chore, docs, refactor, test, ci
- Scope is optional but use it for this project: e.g. `feat(parser): ...`
- Keep subject line under 72 chars

## Versioning
- Use Semantic Versioning: `MAJOR.MINOR.PATCH`
- Keep versions in `Cargo.toml` in sync with Git tags.
- Only update version when explicitly stating or making a release, not for every commit.

## Building Releases

Build targets and flags:

| Target                   | Binary                    | RUSTFLAGS               |
|--------------------------|---------------------------|-------------------------|
| x86_64-unknown-linux-gnu | endspiel-linux-x64        | -C target-cpu=x86-64-v3 |
| aarch64-apple-darwin     | endspiel-mac-arm64        | -C target-cpu=apple-m1  |
| x86_64-pc-windows-msvc   | endspiel-win-x64.exe      | -C target-cpu=x86-64-v3 |
| aarch64-pc-windows-msvc  | endspiel-win-arm64.exe    | (none)                  |

### Local workflow (default)
- Build all targets locally using `cargo build --release --target <target>` with the RUSTFLAGS above.
- Only build the MacOS targets when on a Apple Silicon Mac with the MacOS toolchain installed.
- For Windows targets, use `cargo xwin` (MSVC toolchain) to cross-compile: `cargo xwin build --release --target <target>`.
- Rename binaries as shown in the table above.
- Upload binaries to the GitHub release page via `gh release upload`.
- Create concise release notes summarizing changes since the last release, not just a changelog dump.

### GH Actions workflow (only when explicitly requested)
- Trigger the release workflow via `gh workflow run release.yml`.
- The workflow builds all targets on native runners and creates a GitHub release automatically.

## Testing & Clippy
- before committing always run `cargo test` and `cargo clippy` 