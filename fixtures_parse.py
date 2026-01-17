"""
Parse NTL stream-formatted matrices and random number files into numpy fixtures.
"""
from pathlib import Path
import re
import numpy as np


def parse_ntl_matrix(text: str) -> np.ndarray:
    """
    Parse NTL's mat_ZZ_p << format: lines like [[1 0 1][...]] with brackets.
    """
    # Remove outer brackets and split rows
    text = text.strip()
    # Replace ][ with ]\n[
    text = text.replace("][", "]\n[")
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("[") or not line.endswith("]"):
            continue
        inner = line.strip("[]")
        if not inner:
            continue
        nums = [int(x) for x in inner.split()]
        rows.append(nums)
    return np.array(rows, dtype=object)


def parse_random_numbers(text: str) -> list:
    parts = text.strip().split()
    # first entry is count
    if not parts:
        return []
    count = int(parts[0])
    nums = [int(x) for x in parts[1 : 1 + count]]
    return nums


def parse_kernel_file(path: Path) -> np.ndarray:
    return parse_ntl_matrix(path.read_text())


def parse_random_file(path: Path) -> list:
    return parse_random_numbers(path.read_text())


def save_npz(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, data=arr)


def main():
    kernel_dir = Path("kernel")
    rand_dir = Path("randomNumbers")
    fixtures_dir = Path("Python/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    for kfile in kernel_dir.glob("p_*_*.txt"):
        arr = parse_kernel_file(kfile)
        out = fixtures_dir / (kfile.stem + ".npz")
        save_npz(arr, out)
        print(f"Saved kernel fixture {out} shape={arr.shape}")

    for rfile in rand_dir.glob("p_*_*.txt"):
        nums = parse_random_file(rfile)
        out = fixtures_dir / (rfile.stem + "_rand.txt")
        out.write_text("\n".join(str(x) for x in nums))
        print(f"Saved random fixture {out} count={len(nums)}")


if __name__ == "__main__":
    main()
