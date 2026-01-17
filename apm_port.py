"""
Python port of the GF(p) Las Vegas ECDLP pipeline (prime-field path only).

Implements:
- Finite-field EC arithmetic for a toy secp48k1 curve (y^2 = x^3 + 7 mod p, p ≈ 2^48, p ≡ 1 mod 3).
- Weighted vector generation and unique random-number distribution via MPI files.
- Matrix construction for Las Vegas, modular kernel computation (nullspace), and r-zero DLP extraction.
- Minor-search stack: brute-force 2x2 minors, almost-principal minor search (APM), Schur-complement-style scan, and oblique elimination (LU minor checks).
- MPI-driven loop mirroring the C++ lasVegas() flow (GF(2^m) skipped).

Dependencies: mpi4py, numpy, sympy
Run (example, 2 ranks):
  mpiexec -n 2 python -m mpi4py Python/apm_port.py
"""

import math
import os
import hashlib
import hmac
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from mpi4py import MPI

from combinatorics import (
    nCr,
    init_combinations,
    is_last_combination,
    merge_vectors,
    next_combination,
    kth_combination_indices,
)
from logging_utils import get_logger
from minors import det_mod
from io_inputs import load_dlp_input
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Toy secp48k1 prime generation (p ≡ 1 mod 3, ~2^48)
# ---------------------------------------------------------------------------


def is_prime(n: int) -> bool:
    """
    Deterministic Miller-Rabin for n < 2^61 (sufficient for 48-bit).
    """
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    # write n-1 as 2^r * d
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n == a:
            return True
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def next_prime(n: int) -> int:
    if n < 2:
        return 2
    candidate = n + 1 if n % 2 == 0 else n + 2
    while not is_prime(candidate):
        candidate += 2
    return candidate


def _next_prime_1mod3(n: int) -> int:
    p = next_prime(n)
    while p % 3 != 1:
        p = next_prime(p)
    return p


BITS = 48
OFFSET = 54321
P_SECP48K1 = _next_prime_1mod3(2**BITS + OFFSET)
A_COEFF = 0
B_COEFF = 7

MASTER_NODE = 0

RANDOM_NUM_DIR = Path("randomNumbers")
KERNEL_DIR = Path("kernel")
INPUT_FILE = Path("input/3.txt")

# ---------------------------------------------------------------------------
# Finite field helpers
# ---------------------------------------------------------------------------


def modp(x: int, p: int = P_SECP48K1) -> int:
    return x % p


def modp_inv(x: int, p: int = P_SECP48K1) -> int:
    if x % p == 0:
        raise ZeroDivisionError("inverse of zero")
    return pow(x, -1, p)


# ---------------------------------------------------------------------------
# NTL matrix format helpers (to match C++ stream output)
# ---------------------------------------------------------------------------


def format_ntl_matrix(mat: np.ndarray) -> str:
    """
    Format matrix like NTL's operator<<: [[a b][c d]]
    """
    rows = []
    for r in mat:
        rows.append("[" + " ".join(str(int(x)) for x in r) + "]")
    return "[" + "\n".join(rows) + "]"


def parse_ntl_matrix(text: str) -> np.ndarray:
    s = text.strip()
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return np.zeros((0, 0), dtype=object)
    inner = s[start + 1 : end]
    inner = inner.replace("][", "]\n[")
    rows = []
    for line in inner.splitlines():
        line = line.strip()
        if not (line.startswith("[") and line.endswith("]")):
            continue
        vals = line[1:-1].strip()
        if not vals:
            continue
        rows.append([int(v) for v in vals.split()])
    return np.array(rows, dtype=object)


# ---------------------------------------------------------------------------
# EC over prime field (short Weierstrass, a=0,b=7)
# ---------------------------------------------------------------------------


@dataclass
class Point:
    x: int
    y: int
    infinity: bool = False

    def __iter__(self):
        return iter((self.x, self.y, self.infinity))


class Curve:
    def __init__(self, p: int = P_SECP48K1, a: int = A_COEFF, b: int = B_COEFF):
        self.p = p
        self.a = a % p
        self.b = b % p

    def is_on_curve(self, P: Point) -> bool:
        if P.infinity:
            return True
        x, y = P.x % self.p, P.y % self.p
        return (y * y - (x * x * x + self.a * x + self.b)) % self.p == 0

    def neg(self, P: Point) -> Point:
        if P.infinity:
            return P
        return Point(P.x, (-P.y) % self.p, P.infinity)

    def add(self, P: Point, Q: Point) -> Point:
        if P.infinity:
            return Q
        if Q.infinity:
            return P
        p = self.p
        if P.x % p == Q.x % p and (P.y + Q.y) % p == 0:
            return Point(0, 0, True)

        if P.x == Q.x and P.y == Q.y:
            # tangent slope
            s_num = (3 * P.x * P.x + self.a) % p
            s_den = modp_inv((2 * P.y) % p, p)
        else:
            s_num = (Q.y - P.y) % p
            s_den = modp_inv((Q.x - P.x) % p, p)

        s = (s_num * s_den) % p
        rx = (s * s - P.x - Q.x) % p
        ry = (s * (P.x - rx) - P.y) % p
        return Point(rx, ry, False)

    def scalar_mul(self, k: int, P: Point) -> Point:
        result = Point(0, 0, True)
        addend = P
        while k > 0:
            if k & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            k >>= 1
        return result

    def random_point(self) -> Point:
        while True:
            x = secrets.randbelow(self.p)
            rhs = (x * x * x + self.a * x + self.b) % self.p
            # Try to find a quadratic residue y; sympy.sqrt_mod may be slow, so sample y
            y = pow(rhs, (self.p + 1) // 4, self.p)  # works when p % 4 == 3 (true here)
            if (y * y) % self.p == rhs:
                return Point(x, y, False)


# ---------------------------------------------------------------------------
# Weighted vector generation (matches utils/lasVegas_utils.cpp)
# ---------------------------------------------------------------------------


def generate_weighted_vector(weight: int) -> List[Tuple[int, int, int]]:
    total = ((weight + 1) * (weight + 2)) // 2
    vec = [[0, 0, 0] for _ in range(total)]

    # first column
    tmp_cnt1 = weight
    tmp_cnt3 = weight
    for i in range(total - 1, 0, -1):
        vec[i][0] = tmp_cnt1
        if tmp_cnt1 == 0:
            tmp_cnt3 -= 1
            tmp_cnt1 = tmp_cnt3
        else:
            tmp_cnt1 -= 1

    # second column
    tmp_cnt2 = 0
    tmp_cnt3 = weight
    for i in range(total - 1, 0, -1):
        vec[i][1] = tmp_cnt2
        if tmp_cnt2 == tmp_cnt3:
            tmp_cnt3 -= 1
            tmp_cnt2 = 0
        else:
            tmp_cnt2 += 1

    # third column
    tmp_cnt3 = 1
    tmp_cnt1 = weight + 1
    tmp_cnt2 = 0
    vec[0][2] = weight
    for i in range(total - 1, 0, -1):
        vec[i][2] = tmp_cnt2
        tmp_cnt1 -= 1
        if tmp_cnt1 == 0:
            tmp_cnt2 += 1
            tmp_cnt1 = weight - tmp_cnt3 + 1
            tmp_cnt3 += 1

    return [(a, b, c) for a, b, c in vec]


# ---------------------------------------------------------------------------
# Random numbers (unique per processor set, saved to disk)
# ---------------------------------------------------------------------------


NTL_BITS_PER_LONG = 64 if sys.maxsize > 2**32 else 32


def _derive_key(data: bytes, klen: int = 32) -> bytes:
    # HMAC-SHA256 with zero key to derive intermediate K
    K = hmac.new(b"", data, hashlib.sha256).digest()
    counter = bytearray(8)
    out = bytearray()
    while len(out) < klen:
        out.extend(hmac.new(K, counter, hashlib.sha256).digest())
        for i in range(8):
            counter[i] = (counter[i] + 1) & 0xFF
            if counter[i] != 0:
                break
    return bytes(out[:klen])


def _word_from_bytes(buf: bytes) -> int:
    res = 0
    for b in reversed(buf):
        res = (res << 8) | b
    return res


def _rotl32(x: int, n: int) -> int:
    return ((x << n) & 0xFFFFFFFF) | (x >> (32 - n))


def _chacha20_block(key_words, counter: int, nonce: int) -> bytes:
    # 16-word state
    state = [
        0x61707865,
        0x3320646E,
        0x79622D32,
        0x6B206574,
        *key_words,
        counter & 0xFFFFFFFF,
        (counter >> 32) & 0xFFFFFFFF,
        nonce & 0xFFFFFFFF,
        (nonce >> 32) & 0xFFFFFFFF,
    ]
    working = state[:]
    for _ in range(10):
        # column rounds
        working[0] = (working[0] + working[4]) & 0xFFFFFFFF
        working[12] ^= working[0]
        working[12] = _rotl32(working[12], 16)

        working[8] = (working[8] + working[12]) & 0xFFFFFFFF
        working[4] ^= working[8]
        working[4] = _rotl32(working[4], 12)

        working[0] = (working[0] + working[4]) & 0xFFFFFFFF
        working[12] ^= working[0]
        working[12] = _rotl32(working[12], 8)

        working[8] = (working[8] + working[12]) & 0xFFFFFFFF
        working[4] ^= working[8]
        working[4] = _rotl32(working[4], 7)

        working[1] = (working[1] + working[5]) & 0xFFFFFFFF
        working[13] ^= working[1]
        working[13] = _rotl32(working[13], 16)

        working[9] = (working[9] + working[13]) & 0xFFFFFFFF
        working[5] ^= working[9]
        working[5] = _rotl32(working[5], 12)

        working[1] = (working[1] + working[5]) & 0xFFFFFFFF
        working[13] ^= working[1]
        working[13] = _rotl32(working[13], 8)

        working[9] = (working[9] + working[13]) & 0xFFFFFFFF
        working[5] ^= working[9]
        working[5] = _rotl32(working[5], 7)

        working[2] = (working[2] + working[6]) & 0xFFFFFFFF
        working[14] ^= working[2]
        working[14] = _rotl32(working[14], 16)

        working[10] = (working[10] + working[14]) & 0xFFFFFFFF
        working[6] ^= working[10]
        working[6] = _rotl32(working[6], 12)

        working[2] = (working[2] + working[6]) & 0xFFFFFFFF
        working[14] ^= working[2]
        working[14] = _rotl32(working[14], 8)

        working[10] = (working[10] + working[14]) & 0xFFFFFFFF
        working[6] ^= working[10]
        working[6] = _rotl32(working[6], 7)

        working[3] = (working[3] + working[7]) & 0xFFFFFFFF
        working[15] ^= working[3]
        working[15] = _rotl32(working[15], 16)

        working[11] = (working[11] + working[15]) & 0xFFFFFFFF
        working[7] ^= working[11]
        working[7] = _rotl32(working[7], 12)

        working[3] = (working[3] + working[7]) & 0xFFFFFFFF
        working[15] ^= working[3]
        working[15] = _rotl32(working[15], 8)

        working[11] = (working[11] + working[15]) & 0xFFFFFFFF
        working[7] ^= working[11]
        working[7] = _rotl32(working[7], 7)

        # diagonal rounds
        working[0] = (working[0] + working[5]) & 0xFFFFFFFF
        working[15] ^= working[0]
        working[15] = _rotl32(working[15], 16)

        working[10] = (working[10] + working[15]) & 0xFFFFFFFF
        working[5] ^= working[10]
        working[5] = _rotl32(working[5], 12)

        working[0] = (working[0] + working[5]) & 0xFFFFFFFF
        working[15] ^= working[0]
        working[15] = _rotl32(working[15], 8)

        working[10] = (working[10] + working[15]) & 0xFFFFFFFF
        working[5] ^= working[10]
        working[5] = _rotl32(working[5], 7)

        working[1] = (working[1] + working[6]) & 0xFFFFFFFF
        working[12] ^= working[1]
        working[12] = _rotl32(working[12], 16)

        working[11] = (working[11] + working[12]) & 0xFFFFFFFF
        working[6] ^= working[11]
        working[6] = _rotl32(working[6], 12)

        working[1] = (working[1] + working[6]) & 0xFFFFFFFF
        working[12] ^= working[1]
        working[12] = _rotl32(working[12], 8)

        working[11] = (working[11] + working[12]) & 0xFFFFFFFF
        working[6] ^= working[11]
        working[6] = _rotl32(working[6], 7)

        working[2] = (working[2] + working[7]) & 0xFFFFFFFF
        working[13] ^= working[2]
        working[13] = _rotl32(working[13], 16)

        working[8] = (working[8] + working[13]) & 0xFFFFFFFF
        working[7] ^= working[8]
        working[7] = _rotl32(working[7], 12)

        working[2] = (working[2] + working[7]) & 0xFFFFFFFF
        working[13] ^= working[2]
        working[13] = _rotl32(working[13], 8)

        working[8] = (working[8] + working[13]) & 0xFFFFFFFF
        working[7] ^= working[8]
        working[7] = _rotl32(working[7], 7)

        working[3] = (working[3] + working[4]) & 0xFFFFFFFF
        working[14] ^= working[3]
        working[14] = _rotl32(working[14], 16)

        working[9] = (working[9] + working[14]) & 0xFFFFFFFF
        working[4] ^= working[9]
        working[4] = _rotl32(working[4], 12)

        working[3] = (working[3] + working[4]) & 0xFFFFFFFF
        working[14] ^= working[3]
        working[14] = _rotl32(working[14], 8)

        working[9] = (working[9] + working[14]) & 0xFFFFFFFF
        working[4] ^= working[9]
        working[4] = _rotl32(working[4], 7)

    out = bytearray()
    for i in range(16):
        val = (working[i] + state[i]) & 0xFFFFFFFF
        out.extend(val.to_bytes(4, "little"))
    return bytes(out)


class NTLRandomStream:
    def __init__(self, seed: int):
        if seed < 0:
            seed = -seed
        nb = (seed.bit_length() + 7) // 8
        seed_bytes = seed.to_bytes(nb, "little") if nb else b""
        key = _derive_key(seed_bytes, 32)
        self._key_words = [int.from_bytes(key[i : i + 4], "little") for i in range(0, 32, 4)]
        self._counter = 0
        self._nonce = 0
        self._buf = b""
        self._pos = 0

    def _refill(self) -> None:
        self._buf = _chacha20_block(self._key_words, self._counter, self._nonce)
        self._counter = (self._counter + 1) & 0xFFFFFFFFFFFFFFFF
        self._pos = 0

    def get(self, n: int) -> bytes:
        out = bytearray()
        while n > 0:
            if self._pos >= len(self._buf):
                self._refill()
            take = min(n, len(self._buf) - self._pos)
            out.extend(self._buf[self._pos : self._pos + take])
            self._pos += take
            n -= take
        return bytes(out)

    def random_word(self) -> int:
        nbytes = NTL_BITS_PER_LONG // 8
        buf = self.get(nbytes)
        return _word_from_bytes(buf)


def generate_random_numbers(k: int, ordP: int, stream: NTLRandomStream) -> List[int]:
    nums = []
    while len(nums) < k:
        val = stream.random_word() % ordP
        if val not in nums:
            nums.append(val)
    nums.sort()
    return nums


def save_random_numbers(path: Path, nums: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(f"{len(nums)}\n")
        f.write("\t".join(str(x) for x in nums))


def load_random_numbers(path: Path) -> List[int]:
    with path.open() as f:
        count = int(f.readline().strip())
        parts = f.read().strip().split()
    nums = [int(x) for x in parts[:count]]
    return nums


def generate_random_numbers_for_processors(num_random: int, ordP: int, world_size: int) -> None:
    # Mimic C++ SetSeed(processorId + time(0)) on master before generating all files
    seed_base = int(time.time())
    stream = NTLRandomStream(seed_base)
    for i in range(world_size):
        nums = generate_random_numbers(num_random, ordP, stream)
        bits = ordP.bit_length()
        path = RANDOM_NUM_DIR / f"p_{i}_{world_size}_{bits}.txt"
        save_random_numbers(path, nums)


def get_random_numbers_from_file(rank: int, world_size: int, ordP: int) -> List[int]:
    bits = ordP.bit_length()
    path = RANDOM_NUM_DIR / f"p_{rank}_{world_size}_{bits}.txt"
    return load_random_numbers(path)


def load_kernel_from_file(rank: int, world_size: int, ordP: int) -> np.ndarray:
    bits = ordP.bit_length()
    path = KERNEL_DIR / f"p_{rank}_{world_size}_{bits}.txt"
    lines = path.read_text().splitlines()
    matrix_lines = []
    bracket_balance = 0
    started = False
    for line in lines:
        if not started and "[" in line:
            started = True
        if not started:
            continue
        matrix_lines.append(line.rstrip())
        bracket_balance += line.count("[")
        bracket_balance -= line.count("]")
        if started and bracket_balance == 0:
            break
    matrix_text = "\n".join(matrix_lines).strip()
    return parse_ntl_matrix(matrix_text)


def load_all_kernels(world_size: int, ordP: int) -> List[Tuple[int, np.ndarray]]:
    kernels: List[Tuple[int, np.ndarray]] = []
    for r in range(world_size):
        path = KERNEL_DIR / f"p_{r}_{world_size}_{ordP.bit_length()}.txt"
        if not path.exists():
            continue
        kernels.append((r, load_kernel_from_file(r, world_size, ordP)))
    return kernels


def load_all_kernels(world_size: int, ordP: int) -> List[Tuple[int, np.ndarray]]:
    kernels: List[Tuple[int, np.ndarray]] = []
    for r in range(world_size):
        path = KERNEL_DIR / f"p_{r}_{world_size}_{ordP.bit_length()}.txt"
        if not path.exists():
            continue
        kernels.append((r, load_kernel_from_file(r, world_size, ordP)))
    return kernels


# ---------------------------------------------------------------------------
# Matrix and kernel utilities
# ---------------------------------------------------------------------------


def power_mod(base: int, exp: int, p: int) -> int:
    return pow(base % p, exp, p)


def generate_matrix(curve: Curve, P: Point, Q: Point, k_random: int, t_random: int, random_nums: List[int],
                    weight_vec: List[Tuple[int, int, int]], logger=None) -> Tuple[np.ndarray, bool]:
    """
    Returns (matrix, accident_flag).
    Accident means scalar multiple matched Q (early exit).
    """
    total_rows = k_random + t_random
    cols = len(weight_vec)
    M = np.zeros((total_rows, cols), dtype=object)
    p = curve.p

    # First k_random rows (P multiples)
    for i in range(k_random):
        P1 = curve.scalar_mul(random_nums[i], P)
        if P1.x == Q.x and P1.y == Q.y and not P1.infinity:
            if logger:
                logger.info(f"Accident: P multiple hits Q at row {i}")
            return M, True
        for j, (wx, wy, _) in enumerate(weight_vec):
            M[i, j] = (power_mod(P1.x, wx, p) * power_mod(P1.y, wy, p)) % p

    # Next t_random rows (neg(Q) multiples)
    negQ = curve.neg(Q)
    for idx, i in enumerate(range(k_random, total_rows)):
        P2 = curve.scalar_mul(random_nums[i], negQ)
        if P2.x == Q.x and P2.y == Q.y and not P2.infinity:
            if logger:
                logger.info(f"Accident: -Q multiple hits Q at row {i}")
            return M, True
        for j, (wx, wy, _) in enumerate(weight_vec):
            M[i, j] = (power_mod(P2.x, wx, p) * power_mod(P2.y, wy, p)) % p

    return M % p, False


def modular_nullspace(mat: np.ndarray, p: int, logger=None) -> np.ndarray:
    """
    Compute right nullspace of mat over GF(p), matching NTL::kernel behavior.
    Returns a 2D numpy array with basis vectors as rows (each vector length = mat columns).
    """
    M = mat.copy() % p
    n_rows, n_cols = M.shape

    def gauss_ntl(A: np.ndarray, w: int) -> int:
        n, cols = A.shape
        if w < 0 or w > cols:
            raise ValueError("gauss: bad args")
        l = 0
        for k in range(w):
            if l >= n:
                break
            pos = -1
            for i in range(l, n):
                A[i, k] %= p
                if pos == -1 and A[i, k] % p != 0:
                    pos = i
            if pos != -1:
                if pos != l:
                    A[[pos, l]] = A[[l, pos]]
                piv = pow(int(A[l, k] % p), -1, p)
                piv = (-piv) % p
                for j in range(k + 1, cols):
                    A[l, j] %= p
                for i in range(l + 1, n):
                    t1 = (A[i, k] * piv) % p
                    A[i, k] = 0
                    if t1 != 0:
                        A[i, k + 1 :] = (A[i, k + 1 :] + A[l, k + 1 :] * t1) % p
                l += 1
        return l

    A = M.copy() % p
    r = gauss_ntl(A, n_cols)

    if r == 0:
        ident = np.zeros((n_cols, n_cols), dtype=object)
        for i in range(n_cols):
            ident[i, i] = 1
        return ident

    X = np.zeros((n_cols - r, n_cols), dtype=object)
    if n_cols - r == 0 or n_cols == 0:
        return X

    D = [-1] * n_cols
    inverses = [0] * n_cols
    j = -1
    for i in range(r):
        while True:
            j += 1
            if A[i, j] % p != 0:
                break
        D[j] = i
        inverses[j] = pow(int(A[i, j] % p), -1, p)

    for k in range(n_cols - r):
        v = X[k]
        pos = 0
        for j in range(n_cols - 1, -1, -1):
            if D[j] == -1:
                v[j] = 1 if pos == k else 0
                pos += 1
            else:
                i = D[j]
                t1 = 0
                for s in range(j + 1, n_cols):
                    t1 = (t1 + v[s] * A[i, s]) % p
                v[j] = (-t1 * inverses[j]) % p

    if logger:
        logger.info(f"Nullspace dim={X.shape[0]} cols={X.shape[1]}")
    return X


def is_kernel_having_r_zeros(kernel: np.ndarray, r: int) -> Tuple[bool, int]:
    for i, row in enumerate(kernel):
        zero_cnt = sum(1 for x in row if x == 0)
        if zero_cnt == r:
            return True, i
    return False, -1


def get_dlp(kernel: np.ndarray, row_idx: int, k_random: int, t_random: int, random_nums: List[int], ordP: int) -> int:
    dlp_A = 0
    dlp_B = 0
    for k in range(k_random):
        if kernel[row_idx][k] != 0:
            dlp_A += random_nums[k]
    for k in range(k_random, k_random + t_random):
        if kernel[row_idx][k] != 0:
            dlp_B += random_nums[k]
    dlp_A %= ordP
    dlp_B %= ordP
    if dlp_B == 0:
        raise ZeroDivisionError("dlp_B is zero")
    inv = modp_inv(dlp_B, ordP)
    return (dlp_A * inv) % ordP


def solve_dlp_apm(row_combo: List[int], col_combo: List[int], ker_non_id: np.ndarray, ordP: int, random_nums: List[int]) -> int:
    """
    Port of solveDLP_apm: use row/col combinations of APM to build selector vector and compute DLP.
    ker_non_id: left block of kernel (rows x cols), as saved to disk.
    random_nums: random numbers used to generate the corresponding matrix (from file for that rank).
    """
    org_cols = ker_non_id.shape[1] * 2
    k_rn = (org_cols // 2) - 1
    vec = [True if i <= k_rn else False for i in range(org_cols)]
    for c in col_combo:
        vec[c] = False
    for r in row_combo:
        vec[(org_cols - 1) - r] = True

    rnd = list(random_nums) + [0] * max(0, org_cols - len(random_nums))
    A = 0
    for i in range(k_rn):
        if vec[i]:
            A += rnd[i]
    B = 0
    for i in range(k_rn, org_cols):
        if vec[i]:
            B += rnd[i]
    A %= ordP
    B %= ordP
    if B == 0:
        print("\n Something is wrong... A or B is zero in getDLP() ...")
        return 0
    invB = modp_inv(B, ordP)
    return (A * invB) % ordP


def apm_parallel_3(ordP: int, p_field: int, curve: Optional["Curve"] = None, P: Optional["Point"] = None, Q: Optional["Point"] = None) -> Optional[int]:
    """
    Parity port of principleDeviation_parallel_3:
    - iterate over all kernel files (one per rank)
    - for each block size and 2 deviations, each processor handles a quota of APMs
    - return DLP if a zero determinant APM is found
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world = comm.Get_size()
    logger = get_logger("apm_parallel_3")

    kernels = load_all_kernels(world, ordP)
    kernel_map = {k: mat for k, mat in kernels}
    found_dlp: Optional[int] = None
    for file_id in range(world):
        if file_id not in kernel_map:
            continue
        ker = kernel_map[file_id]
        mat_dim = min(ker.shape[0], ker.shape[1])
        if mat_dim < 2:
            continue
        ker_work = ker[:mat_dim, :mat_dim]
        mat_rows = mat_dim
        number_of_parts = 6
        max_block_size = mat_rows // number_of_parts
        block_start_dims = 2
        number_of_deviations = 2  # fixed as in C++ default

        logger.info(
            f"Processing kernel {file_id}: rows={ker.shape[0]} cols={ker.shape[1]} using_dim={mat_dim} max_block={max_block_size}"
        )

        for block_size in range(block_start_dims, max_block_size + 1):
            complement_size = mat_rows - block_size
            num_apm_half = nCr(complement_size, number_of_deviations)
            num_apm = num_apm_half * num_apm_half
            if num_apm == 0:
                continue

            quota = num_apm // world
            extra_combo = num_apm % world
            if rank == MASTER_NODE:
                quota += extra_combo

            rowCombinationIndex = (rank * quota) // num_apm_half if num_apm_half else 0
            colCombinationIndex = (rank * quota) % num_apm_half if num_apm_half else 0

            rowCombinationVecOrg = kth_combination_indices(complement_size, number_of_deviations, rowCombinationIndex)
            colCombinationVecOrg = kth_combination_indices(complement_size, number_of_deviations, colCombinationIndex)

            block_I = list(range(block_size))
            number_of_PM = mat_rows - block_size
            PM_cnt = 1
            while PM_cnt <= number_of_PM:
                complement_block_I = [i for i in range(mat_rows) if i not in block_I]
                rowCombinationVec = list(rowCombinationVecOrg)
                colCombinationVec = list(colCombinationVecOrg)

                APM_dims = block_size + number_of_deviations
                APM_cnt = 0
                while APM_cnt < quota:
                    row_combo = [0] * APM_dims
                    col_combo = [0] * APM_dims
                    merge_vectors(APM_dims, block_size, number_of_deviations, block_I, complement_block_I, rowCombinationVec, row_combo)
                    merge_vectors(APM_dims, block_size, number_of_deviations, block_I, complement_block_I, colCombinationVec, col_combo)

                    minor = ker_work[np.ix_(row_combo, col_combo)]
                    if det_mod(minor, p_field) == 0:
                        rand_nums = get_random_numbers_from_file(file_id, world, ordP)
                        dlp = solve_dlp_apm(row_combo, col_combo, ker, ordP, rand_nums)
                        logger.info(f"APM zero det: file {file_id} block_size={block_size} PM={PM_cnt} rows={row_combo} cols={col_combo} DLP={dlp}")
                        if curve is not None and P is not None and Q is not None:
                            candidate = int(dlp) % int(ordP)
                            check = curve.scalar_mul(candidate, P)
                            if (check.x % curve.p == Q.x % curve.p) and (check.y % curve.p == Q.y % curve.p) and (check.infinity == Q.infinity):
                                logger.info(f"Verified DLP candidate: {candidate} @ file {file_id} block_size={block_size} PM={PM_cnt}")
                                found_dlp = candidate

                    if not is_last_combination(colCombinationVec, number_of_deviations, complement_size):
                        next_combination(colCombinationVec, number_of_deviations, complement_size)
                    else:
                        if is_last_combination(rowCombinationVec, number_of_deviations, complement_size):
                            break
                        init_combinations(colCombinationVec, number_of_deviations)
                        next_combination(rowCombinationVec, number_of_deviations, complement_size)
                    APM_cnt += 1

                if block_I[-1] < mat_rows - 1:
                    block_I = [x + 1 for x in block_I]
                else:
                    break
                PM_cnt += 1
                comm.Barrier()
                gathered = comm.allgather(found_dlp)
                for val in gathered:
                    if val is not None:
                        found_dlp = val
                        break
                if found_dlp is not None:
                    return found_dlp
    return found_dlp


def save_kernel(kernel: np.ndarray, ordP: int, rank: int, world_size: int) -> None:
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    bits = ordP.bit_length()
    path = KERNEL_DIR / f"p_{rank}_{world_size}_{bits}.txt"
    cols_half = kernel.shape[1] // 2
    non_id = kernel[:, :cols_half]
    matrix_str = format_ntl_matrix(non_id)
    proc_name = MPI.Get_processor_name()
    with path.open("w") as f:
        f.write(matrix_str + "\n")
        f.write(f"{rank}\n")
        f.write(f"{proc_name}\n")
    logger = get_logger("save_kernel")
    logger.info(f"Saved kernel to {path} shape={non_id.shape}")


# ---------------------------------------------------------------------------
# Las Vegas driver (simplified to kernel + r-zero detection path)
# ---------------------------------------------------------------------------


def las_vegas(curve: Curve, P: Point, Q: Point, ordP: int, n_weight: int, offset: int = 1) -> int:
    """
    Port of lasVegas(): kernel generation + APM search (principleDeviation_parallel_3 parity).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    logger = get_logger("lasvegas")

    n = n_weight
    r = 3 * n
    k_random = r - 1
    t_random = r + 1
    mat_rows = k_random + t_random
    weight_vec = generate_weighted_vector(n)

    max_iter_env = os.getenv("APM_MAX_ITER", "")
    if max_iter_env.strip() == "":
        iter_limit = None
    else:
        iter_limit = int(max_iter_env)
        if iter_limit <= 0:
            iter_limit = None
    iteration = 0
    flag = False
    while True:
        dlp = None
        iteration += world_size
        if not flag:
            if rank == MASTER_NODE:
                logger.info(f"Iteration {iteration}: generating random numbers for {world_size} ranks")
                generate_random_numbers_for_processors(mat_rows, ordP, world_size)
            comm.Barrier()

            random_nums = get_random_numbers_from_file(rank, world_size, ordP)
            M, _accident = generate_matrix(curve, P, Q, k_random, t_random, random_nums, weight_vec, logger=logger)

            ker_full = modular_nullspace(M, curve.p, logger=logger)
            if ker_full.shape[0] == 0:
                logger.info(f"rank {rank} kernel empty")
                comm.Barrier()
                continue
            save_kernel(ker_full, ordP, rank, world_size)
            comm.Barrier()

            dlp = apm_parallel_3(ordP, curve.p, curve, P, Q)
            flag = dlp is not None
            if flag:
                logger.info(f"iteration {iteration} DLP found @ rank {rank}")
        else:
            comm.Barrier()
            comm.Barrier()

        comm.Barrier()
        flags = comm.allgather(flag)
        dlps = comm.allgather(dlp if flag else None)
        if any(flags):
            if rank == MASTER_NODE:
                for val in dlps:
                    if val is not None:
                        print(f"[OK] DLP found: {val}")
                        break
            return 0
        if iter_limit is not None and iteration >= iter_limit:
            break

    if rank == MASTER_NODE:
        print("[FAIL] DLP not found within iteration limit")
    return 0


# ---------------------------------------------------------------------------
# Entry point for quick manual runs
# ---------------------------------------------------------------------------


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    logger = get_logger("main")
    curve = Curve()

    if INPUT_FILE.exists():
        inputs = load_dlp_input(INPUT_FILE)
        rec = inputs[0]
        if rec.p != curve.p or rec.a != curve.a or rec.b != curve.b:
            logger.warning(f"Input curve params differ from toy secp48k1; overriding to input p={rec.p}")
            curve = Curve(p=rec.p, a=rec.a, b=rec.b)
        P = Point(rec.Px % curve.p, rec.Py % curve.p, False)
        Q = Point(rec.Qx % curve.p, rec.Qy % curve.p, False)
        ordP = rec.ordP
    else:
        # Default points: generate random P, set Q = kP for random k
        P = curve.random_point()
        k_secret = secrets.randbelow(curve.p)
        Q = curve.scalar_mul(k_secret, P)
        ordP = curve.p  # toy assumption
        if rank == MASTER_NODE:
            logger.info(f"Random k={k_secret}")

    n_weight = ordP.bit_length()  # mirrors C++ NumBits(ordP)

    if rank == MASTER_NODE:
        logger.info(f"Using curve: p={curve.p} (bits={curve.p.bit_length()}, p%3={curve.p % 3}), a={curve.a}, b={curve.b}")
        logger.info(f"P=({P.x},{P.y}), Q=({Q.x},{Q.y}), ordP={ordP}")

    las_vegas(curve, P, Q, ordP, n_weight, offset=1)


if __name__ == "__main__":
    main()
