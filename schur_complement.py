import time
from typing import List

import numpy as np
from mpi4py import MPI

from ntl_compat import gauss_mod, get_modulus
from search_all_two_minors import PartitionData2x2, ResultData2x2, compute_partition_data_2x2, is_2by2_determinant_zero
from apm_port import load_kernel_from_file


def print_partition_data_2x2(pD: PartitionData2x2, processor_id: int) -> None:
    print(f" Processor :: {processor_id} >> i_start :: {pD.i_start}\t j_start :: {pD.j_start}\t quota :: {pD.quota}")


def get_sub_matrix(org_mat: np.ndarray, n: int) -> np.ndarray:
    size = org_mat.shape[0] - n
    mat = np.zeros((size, size), dtype=object)
    for i in range(size):
        for j in range(size):
            mat[i][j] = org_mat[i + n][j + n]
    return mat


def get_sub_matrix_extended(org_mat: np.ndarray, n: int, mat_cols: int | None = None) -> np.ndarray:
    size = org_mat.shape[0] - n
    if mat_cols is None:
        mat_cols = size
    mat = np.zeros((size, mat_cols), dtype=object)
    for i in range(size):
        for j in range(size):
            mat[i][j] = org_mat[i + n][j + n]
    if mat_cols > size:
        mat_col = org_mat.shape[0] - size
        row = 0
        for i in range(org_mat.shape[0] - size, org_mat.shape[0]):
            col = size
            for j in range(org_mat.shape[1] - mat_col, org_mat.shape[1]):
                if col >= mat_cols:
                    break
                mat[row][col] = org_mat[i][j]
                col += 1
            row += 1
    return mat


def bcast_matrix_send(mat: np.ndarray) -> None:
    comm = MPI.COMM_WORLD
    comm.bcast(mat, root=0)


def bcast_matrix_recv() -> np.ndarray:
    comm = MPI.COMM_WORLD
    return comm.bcast(None, root=0)


def send_partition_data_to_slave_processors_2x2(pD: List[PartitionData2x2]) -> None:
    comm = MPI.COMM_WORLD
    for i in range(1, comm.Get_size()):
        comm.send([pD[i].i_start, pD[i].j_start, pD[i].quota], dest=i, tag=0)


def receive_partition_data_from_master_2x2() -> PartitionData2x2:
    comm = MPI.COMM_WORLD
    arr = comm.recv(source=0, tag=0)
    return PartitionData2x2(arr[0], arr[1], arr[2])


def schur_complement(ord_p: int) -> bool:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()
    mod = get_modulus()

    for i in range(1):
        if processor_id == 0:
            ker = load_kernel_from_file(i, total, ord_p)
            if ker.shape[0] != ker.shape[1]:
                print("\n ker row != col ==> using top-left square submatrix\n")
                dim = min(ker.shape[0], ker.shape[1])
                ker = ker[:dim, :dim]
            ker_col_cnt = ker.shape[1]
            column_reduce_constant = 2
            columns_to_be_reduced = ker_col_cnt // column_reduce_constant
            comm.bcast(ker_col_cnt, root=0)
            comm.bcast(column_reduce_constant, root=0)

            column_reduce_count = 1
            org_mat = ker.copy()
            while column_reduce_count < columns_to_be_reduced:
                org_mat = ker.copy()
                org_mat = gauss_mod(org_mat, column_reduce_count, mod)

                h_prime = np.zeros((org_mat.shape[0] - column_reduce_count, org_mat.shape[0]), dtype=object)
                sub = get_sub_matrix(org_mat, column_reduce_count)
                h_prime[:, : sub.shape[1]] = sub
                bcast_matrix_send(h_prime)

                pD = compute_partition_data_2x2(h_prime.shape[0])
                send_partition_data_to_slave_processors_2x2(pD)

                found, _ = is_2by2_determinant_zero(h_prime, pD[0])
                if found:
                    return True
                column_reduce_count += 1
        else:
            ker_col_cnt = comm.bcast(None, root=0)
            column_reduce_constant = comm.bcast(None, root=0)
            columns_to_be_reduced = ker_col_cnt // column_reduce_constant
            column_reduce_count = 1
            while column_reduce_count < columns_to_be_reduced:
                h_prime = bcast_matrix_recv()
                pD = receive_partition_data_from_master_2x2()
                found, _ = is_2by2_determinant_zero(h_prime, pD)
                if found:
                    return True
                column_reduce_count += 1

        comm.Barrier()
        if processor_id == 0:
            print("\n====================================================\n")
    return False
