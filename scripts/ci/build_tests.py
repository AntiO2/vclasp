"""Build once and collect executables/extension from Cargo's structured output."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def build(arguments):
    result = subprocess.run(
        ["cargo", "test", "--locked", "--no-run", "--message-format=json", *arguments],
        stdout=subprocess.PIPE,
        text=True,
    )
    artifacts = []
    for line in result.stdout.splitlines():
        message = json.loads(line)
        if message["reason"] == "compiler-message":
            print(message["message"].get("rendered", ""), end="")
        elif message["reason"] == "compiler-artifact":
            artifacts.append(message)
    result.check_returncode()
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tests = args.output / "rust-tests"
    extension = args.output / "python"
    tests.mkdir()
    extension.mkdir()

    core = build(["--features", "ffmpeg", "--lib", "--test", "rust_session_api"])
    transport = build(["--manifest-path", "object-store-transport/Cargo.toml", "--lib"])

    expected = {"vclasp", "rust_session_api", "vclasp_object_store"}
    copied = set()
    libraries = set()
    for artifact in core + transport:
        name = artifact["target"]["name"]
        if artifact["profile"]["test"] and artifact["executable"]:
            if name not in expected or name in copied:
                raise RuntimeError(f"Unexpected or duplicate test executable: {name}")
            shutil.copy2(artifact["executable"], tests / name)
            copied.add(name)
    for artifact in core:
        if artifact["target"]["name"] == "vclasp" and not artifact["profile"]["test"]:
            libraries.update(p for p in artifact["filenames"] if p.endswith(".so"))
    if copied != expected or len(libraries) != 1:
        raise RuntimeError(f"Incomplete build: tests={copied}, extensions={libraries}")
    shutil.copy2(libraries.pop(), extension / "vclasp.abi3.so")

    # Strip copies only; preserve the incremental Cargo outputs in the build cache.
    for path in [*tests.iterdir(), *extension.iterdir()]:
        subprocess.run(["strip", "--strip-debug", str(path)], check=True)
    archive = args.output / "unit-test-artifacts.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(tests, arcname="rust-tests")
        tar.add(extension, arcname="python")
    print(
        f"Prepared {len(copied)} Rust test executables and one Python extension: {archive}"
    )


if __name__ == "__main__":
    main()
