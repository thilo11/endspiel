#!/usr/bin/env bash
#
# Cross-compile the endspiel UCI engine for 64-bit Android (arm64-v8a).
#
# Produces a standalone CLI binary that speaks UCI over stdin/stdout, runnable
# under Termux or via `adb push` + `adb shell`. At runtime the engine auto-tunes
# its defaults to the device — Hash from /proc/meminfo (~1/16 of available RAM,
# clamped to 16–1024 MB) and Threads from the performance-core count — both still
# overridable via UCI `setoption`.
#
# Requirements:
#   - Android NDK r23+    (set ANDROID_NDK_HOME or ANDROID_NDK_ROOT)
#   - rustup target:      rustup target add aarch64-linux-android
#   - cargo-ndk:          cargo install cargo-ndk
#
# Usage:
#   scripts/build-android.sh                 # API 24, target-cpu=generic
#   API=29 scripts/build-android.sh          # raise the minSdk floor
#   ANDROID_TARGET_CPU=cortex-a76 scripts/build-android.sh   # tune for one device
#
set -euo pipefail

API="${API:-24}"
TARGET="aarch64-linux-android"
ABI="arm64-v8a"
# Default to a portable ARMv8-A baseline so the binary runs on any 64-bit
# Android device (NEON is mandatory on AArch64, so the SIMD eval path is always
# available). Override ANDROID_TARGET_CPU to tune for a specific SoC.
CPU="${ANDROID_TARGET_CPU:-generic}"

if ! command -v cargo-ndk >/dev/null 2>&1; then
  echo "error: cargo-ndk not found. Install it with: cargo install cargo-ndk" >&2
  exit 1
fi

if [[ -z "${ANDROID_NDK_HOME:-}" && -z "${ANDROID_NDK_ROOT:-}" ]]; then
  echo "warning: neither ANDROID_NDK_HOME nor ANDROID_NDK_ROOT is set;" >&2
  echo "         cargo-ndk will try to auto-detect the NDK and may fail." >&2
fi

rustup target add "$TARGET" >/dev/null 2>&1 || true

echo "Building endspiel for $ABI ($TARGET), minSdk $API, target-cpu=$CPU ..."
RUSTFLAGS="-C target-cpu=$CPU" \
  cargo ndk --target "$ABI" --platform "$API" \
  build --release --bin endspiel

OUT="target/${TARGET}/release/endspiel"
echo
echo "Built: $OUT"
file "$OUT" 2>/dev/null || true
echo
echo "Run on a device, e.g.:"
echo "  adb push $OUT /data/local/tmp/endspiel && adb shell chmod +x /data/local/tmp/endspiel"
echo "  adb shell /data/local/tmp/endspiel bench"
