from mpi4py import MPI

from apm_port import load_kernel_from_file
from containment import (
    initCombinations,
    isLastCombination,
    _getNextCombination_continous,
    makeMatrixFromRowColCombination,
    printCombination2,
)
from ntl_compat import determinant


def one_deviation(block_size, block_I, mat, complement_size, complement_block_I):
    processor_id = MPI.COMM_WORLD.Get_rank()
    almost_rows = [0] * (block_size + 1)
    almost_cols = [0] * (block_size + 1)

    for i in range(block_size):
        almost_rows[i] = block_I[i]
        almost_cols[i] = block_I[i]

    for row1_ptr in range(complement_size):
        almost_rows[block_size] = complement_block_I[row1_ptr]
        for col_ptr in range(complement_size):
            flag = False
            if (complement_block_I[row1_ptr] < block_I[0]) and (complement_block_I[col_ptr] < block_I[0]):
                flag = True
            if (complement_block_I[row1_ptr] > block_I[block_size - 1]) and (
                complement_block_I[col_ptr] > block_I[block_size - 1]
            ):
                flag = True

            if flag:
                almost_cols[block_size] = complement_block_I[col_ptr]
                almost = makeMatrixFromRowColCombination(almost_rows, almost_cols, mat)
                if determinant(almost) == 0:
                    printCombination2(almost_rows, block_size + 1)
                    printCombination2(almost_cols, block_size + 1)
                    print("\n apm :: \n", almost)
                    print(f"\n ### determinant is zero (1D) ...\t blockSize :: {block_size}\t @ pId :: {processor_id}")
                    return True
    return False


def two_deviation(block_size, block_I, mat, complement_size, complement_block_I):
    processor_id = MPI.COMM_WORLD.Get_rank()
    almost_rows = [0] * (block_size + 2)
    almost_cols = [0] * (block_size + 2)

    for i in range(block_size):
        almost_rows[i] = block_I[i]
        almost_cols[i] = block_I[i]

    for row1_ptr in range(complement_size):
        almost_rows[block_size] = complement_block_I[row1_ptr]
        for row2_ptr in range(row1_ptr + 1, complement_size):
            if (complement_block_I[row1_ptr] < block_I[0]) and (
                complement_block_I[row2_ptr] > block_I[block_size - 1]
            ):
                continue

            almost_rows[block_size + 1] = complement_block_I[row2_ptr]
            for col1_ptr in range(complement_size):
                almost_cols[block_size] = complement_block_I[col1_ptr]
                for col2_ptr in range(col1_ptr + 1, complement_size):
                    if (complement_block_I[col1_ptr] < block_I[0]) and (
                        complement_block_I[col2_ptr] > block_I[block_size - 1]
                    ):
                        continue

                    if (complement_block_I[row2_ptr] < block_I[0]) and (
                        complement_block_I[col1_ptr] > block_I[block_size - 1]
                    ):
                        continue
                    if (complement_block_I[row1_ptr] > block_I[0]) and (
                        complement_block_I[col2_ptr] < block_I[0]
                    ):
                        continue

                    almost_cols[block_size + 1] = complement_block_I[col2_ptr]
                    almost = makeMatrixFromRowColCombination(almost_rows, almost_cols, mat)
                    if determinant(almost) == 0:
                        printCombination2(almost_rows, block_size + 2)
                        printCombination2(almost_cols, block_size + 2)
                        print(
                            f"\n row1_ptr :: {row1_ptr}\t  block_I[0] :: {block_I[0]}\t row2_ptr :: {row2_ptr}\t block_I[block_size - 1] :: {block_I[block_size - 1]}"
                        )
                        print(f"\n col1  :: {col1_ptr}\t col2 :: {col2_ptr}")
                        print("\n apm :: \n", almost)
                        print(f"\n ### determinant is zero (2D) ...\t blockSize :: {block_size}\t @ pId :: {processor_id}")
                        return True
    return False


def three_deviation(block_size, block_I, mat, complement_size, complement_block_I):
    processor_id = MPI.COMM_WORLD.Get_rank()
    almost_rows = [0] * (block_size + 3)
    almost_cols = [0] * (block_size + 3)

    for i in range(block_size):
        almost_rows[i] = block_I[i]
        almost_cols[i] = block_I[i]

    for row1_ptr in range(complement_size):
        almost_rows[block_size] = complement_block_I[row1_ptr]
        for row2_ptr in range(row1_ptr + 1, complement_size):
            almost_rows[block_size + 1] = complement_block_I[row2_ptr]
            for row3_ptr in range(row2_ptr + 1, complement_size):
                almost_rows[block_size + 2] = complement_block_I[row3_ptr]
                for col1_ptr in range(complement_size):
                    almost_cols[block_size] = complement_block_I[col1_ptr]
                    for col2_ptr in range(col1_ptr + 1, complement_size):
                        almost_cols[block_size + 1] = complement_block_I[col2_ptr]
                        for col3_ptr in range(col2_ptr + 1, complement_size):
                            almost_cols[block_size + 2] = complement_block_I[col3_ptr]
                            almost = makeMatrixFromRowColCombination(almost_rows, almost_cols, mat)
                            if determinant(almost) == 0:
                                print("\n apm :: \n ", almost)
                                print(
                                    f"\n ### determinant is zero (3D) ...\t blockSize :: {block_size}\t @ pId :: {processor_id}"
                                )
                                return True
    return False


def four_deviation(block_size, block_I, mat, complement_size, complement_block_I):
    processor_id = MPI.COMM_WORLD.Get_rank()
    almost_rows = [0] * (block_size + 4)
    almost_cols = [0] * (block_size + 4)

    for i in range(block_size):
        almost_rows[i] = block_I[i]
        almost_cols[i] = block_I[i]

    for row1_ptr in range(complement_size):
        almost_rows[block_size] = complement_block_I[row1_ptr]
        for row2_ptr in range(row1_ptr + 1, complement_size):
            almost_rows[block_size + 1] = complement_block_I[row2_ptr]
            for row3_ptr in range(row2_ptr + 1, complement_size):
                almost_rows[block_size + 2] = complement_block_I[row3_ptr]
                for row4_ptr in range(row3_ptr + 1, complement_size):
                    almost_rows[block_size + 3] = complement_block_I[row4_ptr]
                    for col1_ptr in range(complement_size):
                        almost_cols[block_size] = complement_block_I[col1_ptr]
                        for col2_ptr in range(col1_ptr + 1, complement_size):
                            almost_cols[block_size + 1] = complement_block_I[col2_ptr]
                            for col3_ptr in range(col2_ptr + 1, complement_size):
                                almost_cols[block_size + 2] = complement_block_I[col3_ptr]
                                for col4_ptr in range(col3_ptr + 1, complement_size):
                                    almost_cols[block_size + 3] = complement_block_I[col4_ptr]
                                    # Note: C++ allocates blockSize + 3 here (intentional parity).
                                    row_slice = almost_rows[: block_size + 3]
                                    col_slice = almost_cols[: block_size + 3]
                                    almost = makeMatrixFromRowColCombination(row_slice, col_slice, mat)
                                    if determinant(almost) == 0:
                                        print("\n apm :: \n ", almost)
                                        print(
                                            f"\n ### determinant is zero (4D) ...\t blockSize :: {block_size}\t @ pId :: {processor_id}"
                                        )
                                        printCombination2(almost_rows, block_size + 4)
                                        printCombination2(almost_cols, block_size + 4)
                                        return True
    return False


def four_deviation_segmented(block_size, block_I, mat, complement_size, complement_block_I, dim, row, col):
    processor_id = MPI.COMM_WORLD.Get_rank()
    almost_rows = [0] * (block_size + 4)
    almost_cols = [0] * (block_size + 4)

    for i in range(block_size):
        almost_rows[i] = block_I[i]
        almost_cols[i] = block_I[i]

    first_last_percentage = 22
    first_last_value = (first_last_percentage * complement_size) // 100
    middle_two = (complement_size - (first_last_value * 2)) // 2

    r2_start, r2_end = block_size, first_last_value
    r3_start, r3_end = first_last_value, (r2_end + middle_two)
    r4_start, r4_end = (r2_end + middle_two), (r2_end + middle_two * 2)
    r5_start, r5_end = (r2_end + middle_two * 2), complement_size

    c2_start, c2_end = block_size, first_last_value
    c3_start, c3_end = first_last_value, (r2_end + middle_two)
    c4_start, c4_end = (r2_end + middle_two), (r2_end + middle_two * 2)
    c5_start, c5_end = (r2_end + middle_two * 2), complement_size

    for r2 in range(r2_start, r2_end):
        almost_rows[block_size] = complement_block_I[r2]
        for r3 in range(r3_start + 1, r3_end):
            almost_rows[block_size + 1] = complement_block_I[r3]
            for r4 in range(r4_start + 1, r4_end):
                almost_rows[block_size + 2] = complement_block_I[r4]
                for r5 in range(r5_start + 1, r5_end):
                    almost_rows[block_size + 3] = complement_block_I[r5]
                    for c2 in range(c2_start, c2_end):
                        almost_cols[block_size] = complement_block_I[c2]
                        for c3 in range(c3_start, c3_end):
                            almost_cols[block_size + 1] = complement_block_I[c3]
                            for c4 in range(c4_start, c4_end):
                                almost_cols[block_size + 2] = complement_block_I[c4]
                                for c5 in range(c5_start, c5_end):
                                    almost_cols[block_size + 3] = complement_block_I[c5]
                                    almost = makeMatrixFromRowColCombination(almost_rows, almost_cols, mat)
                                    if determinant(almost) == 0:
                                        print("\n apm :: \n ", almost)
                                        print(
                                            f"\n ### [rPattern-4D] determinant is zero (4D) ...\t PM-size :: {block_size}\t @ pId :: {processor_id}"
                                        )
                                        printCombination2(almost_rows, block_size + 4)
                                        printCombination2(almost_cols, block_size + 4)
                                        for i in range(block_size + 4):
                                            row[i] = almost_rows[i]
                                            col[i] = almost_cols[i]
                                        return True
    return False


def make_all_almost_principle_minors(minor, mat, block_size, block_I, dim, row, col):
    org_rows = mat.shape[0]
    complement_size = org_rows - block_size
    complement_block_I = [0] * complement_size
    cnt = 0
    index = 0
    while cnt < org_rows:
        if cnt != block_I[0]:
            complement_block_I[index] = cnt
            index += 1
            cnt += 1
        else:
            cnt += block_size

    if four_deviation_segmented(block_size, block_I, mat, complement_size, complement_block_I, dim, row, col):
        return True
    return False


def principle_deviation(ord_p):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    mat = load_kernel_from_file(processor_id, total, ord_p)
    mat_rows = mat.shape[0]
    max_block_size = mat_rows // 2

    for block_size in range(2, max_block_size):
        time_s = MPI.Wtime()
        if processor_id == 0:
            print(f"\n Processing block size :: {block_size} of max_blockSize :: {max_block_size}")
        block_I = [0] * block_size
        initCombinations(block_I, block_size)
        dim = 0
        row = [0] * (block_size + 4)
        col = [0] * (block_size + 4)
        while True:
            principle_minor = makeMatrixFromRowColCombination(block_I, block_I, mat)
            if make_all_almost_principle_minors(principle_minor, mat, block_size, block_I, dim, row, col):
                return True
            if not isLastCombination(block_I, block_size, max_block_size):
                _getNextCombination_continous(block_I, max_block_size, block_size)
            else:
                break
        time_e = MPI.Wtime()
        print(f"\n blockSize :: {block_size}\t Time :: {(time_e - time_s)} Sec.\t pId :: {processor_id}")
    return False


oneDeviation = one_deviation
twoDeviation = two_deviation
threeDeviation = three_deviation
fourDeviation = four_deviation
fourDeviation_segmented = four_deviation_segmented
makeAllAlmostPrincipleMinors = make_all_almost_principle_minors
principleDeviation = principle_deviation
