"""Check the native codec toolchain used to build VClasp."""

import argparse
import ctypes
import pathlib
import subprocess
import sys


EXPECTED = {
    "libavcodec": 61,
    "libavformat": 61,
    "libavutil": 59,
    "libswscale": 8,
}


def command(*args: str) -> str:
    result = subprocess.run(args, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def check() -> None:
    for library, major in EXPECTED.items():
        version = command("pkg-config", "--modversion", library)
        if int(version.split(".", 1)[0]) != major:
            raise RuntimeError(
                f"{library} {version}: expected major {major} (FFmpeg 7.1)"
            )
        print(f"{library}: {version}")

    x264_version = command("pkg-config", "--modversion", "x264")
    x264_build = int(x264_version.split(".")[1])
    x264_dir = pathlib.Path(command("pkg-config", "--variable=libdir", "x264"))
    x264_lib = x264_dir / "libx264.so"
    library = ctypes.CDLL(str(x264_lib))
    symbol = f"x264_encoder_open_{x264_build}"
    if not hasattr(library, symbol):
        raise RuntimeError(
            f"{x264_lib} does not export {symbol}; header/library mismatch"
        )
    print(f"x264: {x264_version}, {x264_lib}, {symbol}")

    ffmpeg = command("ffmpeg", "-version").splitlines()[0]
    if not ffmpeg.startswith("ffmpeg version 7.1"):
        raise RuntimeError(f"expected FFmpeg 7.1 executable on PATH, found {ffmpeg}")
    print(ffmpeg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-extension", action="store_true")
    args = parser.parse_args()
    try:
        check()
        if args.import_extension:
            import vclasp

            print(f"Python extension: {vclasp.__file__}")
    except (
        FileNotFoundError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
        RuntimeError,
    ) as error:
        print(f"Native toolchain check failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
