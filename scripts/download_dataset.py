import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, target)
    print(f"Saved to: {target}")


def extract_if_archive(file_path: Path, output_dir: Path) -> None:
    if file_path.suffix.lower() == ".zip":
        print(f"Extracting zip: {file_path}")
        with zipfile.ZipFile(file_path, "r") as zf:
            zf.extractall(output_dir)
        print(f"Extracted to: {output_dir}")


def copy_if_not_archive(file_path: Path, output_dir: Path) -> None:
    if file_path.suffix.lower() != ".zip":
        output_dir.mkdir(parents=True, exist_ok=True)
        dst = output_dir / file_path.name
        shutil.copy2(file_path, dst)
        print(f"Copied file to: {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download public encrypted-traffic dataset files from direct URLs"
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Direct downloadable URL. Repeat this option for multiple files.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save downloaded raw files.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("data/extracted"),
        help="Directory to extract archives.",
    )
    args = parser.parse_args()

    if not args.url:
        print("No URL provided. Please pass at least one --url argument.")
        return

    for idx, url in enumerate(args.url, start=1):
        file_name = url.rstrip("/").split("/")[-1] or f"dataset_{idx}.bin"
        target = args.raw_dir / file_name
        download_file(url, target)
        extract_if_archive(target, args.extract_dir)
        copy_if_not_archive(target, args.extract_dir)

    print("Done. You can now run pcap preprocessing on files in data/extracted.")


if __name__ == "__main__":
    main()
