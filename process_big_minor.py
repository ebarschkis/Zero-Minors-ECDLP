from pathlib import Path

from mpi4py import MPI

from ntl_compat import gauss_mod, set_modulus
from schur_complement_serial import make_kernel_from_matrix
from schur_complement import get_sub_matrix_extended
from containment import processAllSubMatricesOfDimension_partWise


def process_bigger_minors() -> None:
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    p = 33554393
    set_modulus(p)
    file_name = Path(f"kernel_DB/25_29/25/kernel_c1_25_{processor_id + 1}.txt")
    print(f"\n fileName :: {file_name}")

    if not file_name.exists():
        print(f"\n Error opening file :: {file_name}")
        return

    text = file_name.read_text().strip()
    from apm_port import parse_ntl_matrix

    org_mat = parse_ntl_matrix(text)
    ext_org_mat = make_kernel_from_matrix(org_mat, p)
    max_reduce_cnt = org_mat.shape[1]

    columns_reduce_cnt = 0
    while columns_reduce_cnt < max_reduce_cnt:
        ker = ext_org_mat.copy()
        ker = gauss_mod(ker, columns_reduce_cnt, p)
        print(f"\n col reduce cnt :: {columns_reduce_cnt}")
        h_prime = get_sub_matrix_extended(ker, columns_reduce_cnt)

        dimension = h_prime.shape[0] - 3
        while True:
            print("\n [S]-######################################################## \n")
            out_name = Path(f"output/dimension_{dimension}_0_{columns_reduce_cnt}.txt")
            out_name.parent.mkdir(parents=True, exist_ok=True)
            with out_name.open("w") as fout:
                fout.write(f"Processing File :: {file_name}\n")
                print(f"Processing dimension :: {dimension}")
                processAllSubMatricesOfDimension_partWise(dimension, h_prime, fout)
            dimension -= 1
            print("\n [E]-######################################################## \n")
            break
        columns_reduce_cnt += 1
        break
    columns_reduce_cnt += 1
