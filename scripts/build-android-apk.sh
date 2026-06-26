#!/usr/bin/env bash
#
# Build the Endspiel Open Exchange (OEX) engine APK for arm64 Android.
#
# This wraps the cross-compiled UCI binary in a tiny APK that chess GUIs
# (DroidFish, Chess for Android, …) discover via the Open Exchange protocol.
# Unlike the bare ELF from build-android.sh, the APK installs like a normal app
# on a non-rooted device — no Termux or `adb` required.
#
# The native binary is dropped in as jniLibs/arm64-v8a/libendspiel.so. Naming it
# lib*.so plus android:extractNativeLibs=true makes the package installer extract
# it into the app's nativeLibraryDir with the execute bit set, which is the only
# place a non-rooted device may exec a binary.
#
# Requirements:
#   - The arm64 binary, built first via scripts/build-android.sh, OR pass its
#     path as $1 / $ENDSPIEL_ANDROID_BIN.
#   - JDK 17+ and the Android SDK (platform-34, build-tools). Set ANDROID_HOME.
#   - Gradle 8.x on PATH (or a wrapper in android/oex).
#
# Usage:
#   scripts/build-android.sh                 # produce the arm64 binary
#   scripts/build-android-apk.sh             # then wrap it into an APK
#   scripts/build-android-apk.sh path/to/endspiel   # use an explicit binary
#
# Output: android/oex/app/build/outputs/apk/release/app-release-unsigned.apk
# (CI signs it; for a local install, sign with apksigner — see android/oex/README.md)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OEX_DIR="$ROOT/android/oex"

BIN="${1:-${ENDSPIEL_ANDROID_BIN:-$ROOT/target/aarch64-linux-android/release/endspiel}}"
if [[ ! -f "$BIN" ]]; then
  echo "error: Android binary not found at: $BIN" >&2
  echo "       Build it first with: scripts/build-android.sh" >&2
  exit 1
fi

JNI_DIR="$OEX_DIR/app/src/main/jniLibs/arm64-v8a"
mkdir -p "$JNI_DIR"
cp "$BIN" "$JNI_DIR/libendspiel.so"
echo "Staged $(du -h "$JNI_DIR/libendspiel.so" | cut -f1) -> jniLibs/arm64-v8a/libendspiel.so"

# Version metadata: pass VERSION_NAME / VERSION_CODE from CI; fall back to Cargo.
VNAME="${VERSION_NAME:-$(grep -m1 '^version' "$ROOT/Cargo.toml" | sed -E 's/.*"([^"]+)".*/\1/')}"
VCODE="${VERSION_CODE:-1}"

GRADLE="${GRADLE:-gradle}"
if [[ -x "$OEX_DIR/gradlew" ]]; then
  GRADLE="$OEX_DIR/gradlew"
fi

echo "Building APK (versionName=$VNAME versionCode=$VCODE) ..."
( cd "$OEX_DIR" && "$GRADLE" --no-daemon -Pvname="$VNAME" -Pvcode="$VCODE" assembleRelease )

APK="$OEX_DIR/app/build/outputs/apk/release/app-release-unsigned.apk"
echo
echo "Built: $APK"
echo "Sign it before installing, e.g.:"
echo "  zipalign -p 4 \"$APK\" endspiel-aligned.apk"
echo "  apksigner sign --ks <keystore.jks> --out endspiel-android-arm64.apk endspiel-aligned.apk"
