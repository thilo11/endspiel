# Endspiel — Open Exchange (OEX) engine APK

This is a thin Android wrapper that packages the Endspiel UCI engine as an
**Open Exchange engine plugin**. Chess GUIs that speak the Open Exchange
protocol — [DroidFish](https://github.com/peterosterlund2/droidfish),
[Chess for Android](https://www.aartbik.com/MISC/android.html), and others —
auto-discover it after install and let you pick **Endspiel** as a UCI engine.

Unlike the bare ELF produced by `scripts/build-android.sh`, this installs like a
normal app on a non-rooted device: **no Termux, no `adb`**.

## How it works

The engine has no UI of its own. The protocol only needs:

- an `Activity` advertising the `intent.chess.provider.ENGINE` action plus a
  `chess.provider.engine.authority` meta-data pointing at a `ContentProvider`;
- `res/xml/enginelist.xml`, mapping an engine `name` to a `filename` and target ABI;
- the UCI binary shipped at `jniLibs/arm64-v8a/libendspiel.so`.

Naming the binary `lib*.so` together with `android:extractNativeLibs="true"`
makes the package installer drop it into the app's `nativeLibraryDir` with the
execute bit set — the one location a non-rooted device may `exec()` a binary.
The GUI reads that path (directly from `nativeLibraryDir`, or via the
`ContentProvider` fallback) and speaks UCI to it over stdin/stdout. The NNUE net
is embedded in the binary, so the APK is self-contained.

The `com.kalab.chess.enginesupport.ChessEngineProvider` class is vendored from
Gerhard Kalab's [Chess Engine Support Library](https://github.com/gkalab/chessenginesupport-androidlib)
(Apache-2.0).

## Building

The APK is built by the release workflow (`.github/workflows/release.yml`); it
needs no committed `jniLibs`. To build it by hand you need JDK 17+, the Android
SDK (platform-34 + build-tools), and Gradle 8.x:

```bash
scripts/build-android.sh         # cross-compile the arm64 UCI binary (NDK)
scripts/build-android-apk.sh     # stage the binary and assemble the APK
```

That emits an **unsigned** release APK at
`app/build/outputs/apk/release/app-release-unsigned.apk`. Android refuses to
install unsigned APKs, so sign it (a self-signed key is fine for sideloading):

```bash
keytool -genkeypair -keystore endspiel.jks -alias endspiel \
  -keyalg RSA -keysize 2048 -validity 10000 \
  -storepass changeit -keypass changeit -dname "CN=Endspiel"

zipalign -p 4 app/build/outputs/apk/release/app-release-unsigned.apk aligned.apk
apksigner sign --ks endspiel.jks --ks-pass pass:changeit \
  --out endspiel-android-arm64.apk aligned.apk
```

### Stable upgrade signatures in CI

Android keys an app's upgrade path to its signing certificate. The release
workflow signs with a persistent keystore **if** these repository secrets are
set, so users can upgrade in place:

- `ANDROID_KEYSTORE_BASE64` — `base64 -w0 endspiel.jks`
- `ANDROID_KEYSTORE_PASSWORD`
- `ANDROID_KEY_ALIAS`
- `ANDROID_KEY_PASSWORD`

If they are absent, CI signs with an **ephemeral** key so the artifact is still
installable for testing — but each release then has a different signature, so
users must uninstall the previous build before installing a new one.

## Installing

Download `endspiel-android-arm64.apk` from the GitHub release, copy it to the
device, and open it (enable "install from unknown sources" if prompted). Then in
your chess GUI, add a UCI engine and select **Endspiel** from the list.
