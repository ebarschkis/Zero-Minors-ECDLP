import os
import time
from pathlib import Path

from mpi4py import MPI

from apm_port import KERNEL_DIR, load_kernel_from_file
from combinatorics import nCr
from containment import get_kth_combination, processAllSubMatricesOfDimension_parallel


def brute_force_all_minor_parallel(ord_p: int) -> bool:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    for file_id in range(total):
        kernel_path = KERNEL_DIR / f"p_{file_id}_{total}_{ord_p.bit_length()}.txt"
        if not kernel_path.exists():
            if processor_id == 0:
                print(f"\n Skipping missing kernel file :: {kernel_path}")
            continue
        mat = load_kernel_from_file(file_id, total, ord_p)
        print(f"\n Processing File :: {file_id}\t mat.r :: {mat.shape[0]}\t mat.c :: {mat.shape[1]}")

        mat_dim = min(mat.shape[0], mat.shape[1])
        if mat_dim < 2:
            continue
        mat = mat[:mat_dim, :mat_dim]
        mat_rows = mat_dim
        max_block_size = mat_rows - 1

        disable_output = os.environ.get("APM_DISABLE_OUTPUT", "").lower() in {"1", "true", "yes"}
        out_dir = Path(os.environ.get("APM_OUTPUT_DIR", "output/tmp"))
        if disable_output:
            fout = open(os.devnull, "w")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            file_name = out_dir / f"ZM_{processor_id}_fId_{file_id}.txt"
            fout = file_name.open("w")
            fout.write(f"Processing File :: {file_name}\n")

        try:
            for block_size in range(2, max_block_size):
                time_s = time.time()
                print(f" Processing block size :: {block_size} of {max_block_size}", end="")

                total_row_combinations = nCr(max_block_size, block_size)
                total_combinations = total_row_combinations * total_row_combinations
                quota = total_combinations // total
                extra = total_combinations % total

                symbol_vec = list(range(max_block_size))
                row_combination_index = (processor_id * quota) // total_row_combinations
                col_combination_index = (processor_id * quota) % total_row_combinations

                row_combination_vec = [0] * block_size
                col_combination_vec = [0] * block_size
                get_kth_combination(symbol_vec, max_block_size, block_size, int(row_combination_index), row_combination_vec)
                get_kth_combination(symbol_vec, max_block_size, block_size, int(col_combination_index), col_combination_vec)

                if processor_id == total - 1:
                    quota += extra

                processAllSubMatricesOfDimension_parallel(block_size, mat, fout, row_combination_vec, col_combination_vec, int(quota))
                comm.Barrier()
                time_e = time.time()
                print(f"\t Time :: {time_e - time_s} seconds.\n")
        finally:
            fout.close()

    return True
