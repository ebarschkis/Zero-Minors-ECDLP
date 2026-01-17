from pathlib import Path

from mpi4py import MPI

from apm_port import parse_ntl_matrix
from combinatorics import nCr
from containment import get_kth_combination, processAllSubMatricesOfDimension_parallel
from ntl_compat import gauss_mod, set_modulus
from schur_complement import get_sub_matrix_extended
from schur_complement_serial import make_kernel_from_matrix


def process_big_minors_parallel() -> None:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    p = 33554393
    set_modulus(p)

    file_id = 1
    while file_id <= 10:
        file_name = Path(f"kernel_DB/25_29/25/kernel_c1_25_{file_id}.txt")
        if processor_id == 0:
            print(f"\n fileName :: {file_name}")

        if not file_name.exists():
            print(f"\n Error opening file :: {file_name}")
            return

        org_mat = parse_ntl_matrix(file_name.read_text())
        ext_org_mat = make_kernel_from_matrix(org_mat, p)

        max_reduce_cnt = org_mat.shape[1]
        if processor_id == 0:
            print(f" Mat.Row :: {org_mat.shape[0]}\t Mat.Col :: {org_mat.shape[1]}")

        columns_reduce_cnt = 0
        while columns_reduce_cnt < max_reduce_cnt:
            ker = ext_org_mat.copy()
            ker = gauss_mod(ker, columns_reduce_cnt, p)
            if processor_id == 0:
                print(f"\n col reduce cnt :: {columns_reduce_cnt}")
            h_prime = get_sub_matrix_extended(ker, columns_reduce_cnt)

            dimension = h_prime.shape[0] - 3
            while dimension > 65:
                if processor_id == 0:
                    print("\n [S]-######################################################## \n")
                out_name = Path(f"output/prime_25_d_{dimension}_pId_{processor_id}_fId_{file_id}.txt")
                out_name.parent.mkdir(parents=True, exist_ok=True)
                with out_name.open("w") as fout:
                    fout.write(f"Processing File :: {file_name}\n")
                    if processor_id == 0:
                        print(f" Processing dimension :: {dimension}")

                    total_row_combinations = nCr(h_prime.shape[0], dimension)
                    total_combinations = total_row_combinations * total_row_combinations
                    quota = total_combinations // total

                    row_combination_number = (processor_id * quota) // total_row_combinations
                    col_combination_number = (processor_id * quota) % total_row_combinations
                    if processor_id == total - 1:
                        quota += total_row_combinations - (quota * total)

                    symbol_vec = list(range(h_prime.shape[0]))
                    row_combination_vec = [0] * dimension
                    col_combination_vec = [0] * dimension
                    get_kth_combination(symbol_vec, h_prime.shape[0], dimension, int(row_combination_number), row_combination_vec)
                    get_kth_combination(symbol_vec, h_prime.shape[0], dimension, int(col_combination_number), col_combination_vec)

                    processAllSubMatricesOfDimension_parallel(dimension, h_prime, fout, row_combination_vec, col_combination_vec, int(quota))

                dimension -= 1
                comm.Barrier()
                if processor_id == 0:
                    print("\n [E]-######################################################## \n")
                break
            columns_reduce_cnt += 1
        columns_reduce_cnt += 1
        file_id += 1
