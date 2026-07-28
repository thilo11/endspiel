#!/usr/bin/env bash

set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "error: Pi 5 PGO training must run natively on AArch64" >&2
  exit 1
fi

repo_root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
output="${1:-${repo_root}/target/pi5-pgo/endspiel}"
build_root="${repo_root}/target/pi5-pgo-build"
temp_root="${TMPDIR:-/tmp}"
profile_dir="$(mktemp -d "${temp_root%/}/endspiel-pi5-pgo.XXXXXX")"

cleanup() {
  rm -rf "${profile_dir}"
}
trap cleanup EXIT

export CARGO_INCREMENTAL=0
export CARGO_PROFILE_RELEASE_LTO=fat

rustflags="-C target-cpu=cortex-a76"
instrumented_target="${build_root}/instrumented"
optimized_target="${build_root}/optimized"
merged_profile="${profile_dir}/merged.profdata"

cd "${repo_root}"

echo "Building Cortex-A76 instrumented binary..."
CARGO_TARGET_DIR="${instrumented_target}" \
  RUSTFLAGS="${rustflags} -Cprofile-generate=${profile_dir}" \
  cargo build --release --locked --bin endspiel

echo "Training profile with the built-in benchmark..."
"${instrumented_target}/release/endspiel" bench

llvm_profdata="$(
  find "$(rustc --print sysroot)" -type f -name llvm-profdata -print -quit
)"
if [[ -z "${llvm_profdata}" ]]; then
  echo "error: llvm-profdata not found; install llvm-tools-preview" >&2
  exit 1
fi

"${llvm_profdata}" merge \
  -o "${merged_profile}" \
  "${profile_dir}"

echo "Building Cortex-A76 PGO binary..."
CARGO_TARGET_DIR="${optimized_target}" \
  RUSTFLAGS="${rustflags} -Cprofile-use=${merged_profile}" \
  cargo build --release --locked --bin endspiel

install -Dm755 "${optimized_target}/release/endspiel" "${output}"
echo "Pi 5 PGO binary: ${output}"
