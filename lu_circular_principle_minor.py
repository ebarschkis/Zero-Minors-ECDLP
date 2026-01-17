from pathlib import Path

from mpi4py import MPI

from apm_port import parse_ntl_matrix
from circular_swap import circular_shift_matrix_row
from ntl_compat import determinant, set_modulus


def get_kth_principle_minor(mat, k):
    return mat[:k, :k]


def check_all_principle_minors(mat):
    k = 1
    while k < mat.shape[0]:
        minor = get_kth_principle_minor(mat, k)
        if determinant(minor) == 0:
            print(f"\n k :: {k}")
            print("\n minor :: \n", minor)
            print(" determinant :: ", determinant(minor))
        k += 1


def LU_Circular_PrincipleMinorTest():
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    p = 536870909
    set_modulus(p)

    file_id = 1
    while file_id < 10:
        file_name = Path(f"kernel_DB/25_29/29/kernel_c1_29_{file_id}.txt")
        if processor_id == 0:
            print(f" fileName :: {file_name}")

        if not file_name.exists():
            print(f"\n Error opening file :: {file_name}")
            return

        ker = parse_ntl_matrix(file_name.read_text())

        row_cnt = 0
        while row_cnt < ker.shape[0]:
            ker = circular_shift_matrix_row(ker)
            check_all_principle_minors(ker)
            row_cnt += 1
        file_id += 1


get_kth_principleMinor = get_kth_principle_minor
checkAll_principleMinors = check_all_principle_minors
