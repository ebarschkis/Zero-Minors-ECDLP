import itertools
import time
from typing import Tuple

import numpy as np

from linear_algebra import gauss, determinant_mod
from combinatorics import (
    init_combinations,
    is_last_combination,
    kth_combination_indices,
    nCr,
    next_combination,
)
from partitioning import partition_linear
from logging_utils import get_logger


def make_submatrix(mat: np.ndarray, rows: list, cols: list) -> np.ndarray:
    sub = mat[np.ix_(rows, cols)]
    return sub


def det_mod(sub: np.ndarray, p: int) -> int:
    return determinant_mod(sub, p)


def brute_force_2x2(mat: np.ndarray, p: int) -> Tuple[bool, Tuple[int, int, int, int]]:
    """
    Scan all 2x2 minors; returns (found, (r1,r2,c1,c2)).
    """
    rows = range(mat.shape[0])
    cols = range(mat.shape[1])
    for r1, r2 in itertools.combinations(rows, 2):
        for c1, c2 in itertools.combinations(cols, 2):
            sub = mat[np.ix_([r1, r2], [c1, c2])]
            if det_mod(sub, p) == 0:
                return True, (r1, r2, c1, c2)
    return False, (-1, -1, -1, -1)


def distributed_bruteforce_2x2(mat: np.ndarray, p: int, rank: int, world: int) -> Tuple[bool, Tuple[int, int, int, int]]:
    """
    Partition the 2x2 search by distributing row-pair indices across ranks.
    """
    logger = get_logger("bf2x2")
    row_pairs = list(itertools.combinations(range(mat.shape[0]), 2))
    chunk = (len(row_pairs) + world - 1) // world
    start = rank * chunk
    end = min(len(row_pairs), start + chunk)
    for rp_idx in range(start, end):
        r1, r2 = row_pairs[rp_idx]
        for c1, c2 in itertools.combinations(range(mat.shape[1]), 2):
            sub = mat[np.ix_([r1, r2], [c1, c2])]
            if det_mod(sub, p) == 0:
                logger.info(f"Zero 2x2 minor at rows {r1},{r2} cols {c1},{c2}")
                return True, (r1, r2, c1, c2)
    return False, (-1, -1, -1, -1)


def apm_search(
    ker: np.ndarray,
    p: int,
    ordP: int,
    rank: int,
    world: int,
    block_start: int = 2,
    deviations_start: int = 2,
    deviations_end: int = 2,
) -> Tuple[bool, dict]:
    """
    Port of principleDeviation_parallel_3: search almost-principal minors (APM) with MPI partitioning.
    C++ default looks only at 2 deviations; use deviations_end=2 for parity.
    """
    logger = get_logger("apm")
    n_rows = ker.shape[0]
    number_of_parts = 6
    max_block = n_rows // number_of_parts

    for block_size in range(block_start, max_block + 1):
        for num_dev in range(deviations_start, deviations_end + 1):
            complement_size = n_rows - block_size
            num_combo_half = nCr(complement_size, num_dev)
            total_apm = num_combo_half * num_combo_half
            part = partition_linear(total_apm, rank, world)
            start_idx, end_idx = part["start"], part["end"]

            # initialize block_I contiguous starting at 0
            block_I = list(range(block_size))
            number_of_PM = n_rows - block_size
            PM_cnt = 1
            while PM_cnt <= number_of_PM:
                complement = [i for i in range(n_rows) if i not in block_I]
                for linear_idx in range(start_idx, end_idx):
                    row_idx = linear_idx // num_combo_half
                    col_idx = linear_idx % num_combo_half
                    row_dev_vec = kth_combination_indices(complement_size, num_dev, row_idx)
                    col_dev_vec = kth_combination_indices(complement_size, num_dev, col_idx)

                    row_combo = block_I + [complement[i] for i in row_dev_vec]
                    col_combo = block_I + [complement[i] for i in col_dev_vec]
                    minor = make_submatrix(ker, row_combo, col_combo)
                    if det_mod(minor, p) == 0:
                        logger.info(
                            f"APM zero det block_size={block_size} num_dev={num_dev} PM_cnt={PM_cnt} rows={row_combo} cols={col_combo}"
                        )
                        return True, {
                            "block_size": block_size,
                            "num_dev": num_dev,
                            "rows": row_combo,
                            "cols": col_combo,
                            "PM_cnt": PM_cnt,
                        }
                # move to next principal minor (contiguous block)
                if not is_last_combination(block_I, block_size, n_rows):
                    block_I = [x + 1 for x in block_I]
                else:
                    break
                PM_cnt += 1
    return False, {}
