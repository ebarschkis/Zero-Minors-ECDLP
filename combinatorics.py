from typing import List


def init_combinations(vec: List[int], k: int) -> None:
    for i in range(k):
        vec[i] = i


def is_last_combination(vec: List[int], k: int, n: int) -> bool:
    start = n - k
    for i in range(k):
        if vec[i] != start + i:
            return False
    return True


def next_combination(vec: List[int], k: int, n: int) -> bool:
    """
    Lexicographic next combination (in-place). Returns False if at last combination.
    """
    for i in reversed(range(k)):
        if vec[i] != i + n - k:
            vec[i] += 1
            for j in range(i + 1, k):
                vec[j] = vec[j - 1] + 1
            return True
    return False


def kth_combination_indices(n: int, k: int, idx: int) -> List[int]:
    """
    Return the idx-th combination (0-based) of size k from range(n) in lexicographic order.
    """
    out = []
    remaining = idx
    next_val = 0
    for i in range(k, 0, -1):
        for v in range(next_val, n):
            count = nCr(n - v - 1, i - 1)
            if remaining < count:
                out.append(v)
                next_val = v + 1
                break
            remaining -= count
    return out


def get_kth_combination(symbols: List[int], n: int, k: int, idx: int, out: List[int]) -> None:
    indices = kth_combination_indices(n, k, idx)
    for i, v in enumerate(indices):
        out[i] = symbols[v]


def nCr(n: int, r: int) -> int:
    if r < 0 or r > n:
        return 0
    r = min(r, n - r)
    num = 1
    den = 1
    for i in range(1, r + 1):
        num *= n - r + i
        den *= i
    return num // den


def factorial(n: int) -> int:
    if n == 0:
        return 1
    return n * factorial(n - 1)


def factorial_iterative(n: int) -> int:
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res


def merge_vectors(total: int, block_size: int, devs: int, block: List[int], complement: List[int], dev_vec: List[int], out: List[int]) -> None:
    """
    Merge contiguous block indices with deviation indices selected from complement.
    """
    b_ptr = 0
    d_ptr = 0
    for i in range(total):
        if b_ptr < block_size and (d_ptr >= devs or block[b_ptr] <= complement[dev_vec[d_ptr]]):
            out[i] = block[b_ptr]
            b_ptr += 1
        else:
            out[i] = complement[dev_vec[d_ptr]]
            d_ptr += 1
