import random

from mpi4py import MPI

from combinatorics import nCr, get_kth_combination
from containment import initCombinations, isLastCombination, _getNextCombination
from ntl_compat import get_modulus


def get_random_unique_vec(vector_size, e):
    p = get_modulus()
    vec = [0] * vector_size
    random_cnt = 0
    while random_cnt < vector_size:
        tmp = random.randrange(p)
        if tmp == 0:
            continue
        if tmp == e % p:
            continue
        flag = True
        for i in range(random_cnt):
            if vec[i] == tmp:
                flag = False
                break
        if flag:
            vec[random_cnt] = tmp
            random_cnt += 1
    return vec


def sub_vec_addition(P, Q, vec, chose, EC):
    combo_vec = [0] * chose
    initCombinations(combo_vec, chose)

    while True:
        total = 0
        for i in range(chose):
            total += vec[combo_vec[i]]
        total %= get_modulus()

        tmp_point = type(P)()
        EC.scalarMultiplicationDA(P, total, tmp_point)

        if tmp_point.x == Q.x and tmp_point.y == Q.y:
            print(
                f"\n sum :: {total}\t chose :: {chose}\t vec-Size :: {len(vec)}"
            )
            print(
                f"\n tmpPt :: {tmp_point.x}:{tmp_point.y}\t Q ::{Q.x}:{Q.y}"
            )
            return True

        if not isLastCombination(combo_vec, chose, len(vec)):
            _getNextCombination(combo_vec, len(vec), chose)
        else:
            break
    return False


def hypothesis_A(P, Q, ord_p, p_bits, offset, EC):
    processor_id = MPI.COMM_WORLD.Get_rank()
    print(f"\n in hypothesis_A :: ordP :: {ord_p}")

    vector_start_size = 19
    vector_end_size = 24

    for vector_size in range(vector_start_size, vector_end_size):
        print(f"\n Processing vectorSize :: {vector_size}")
        cnt = 1000
        for _ in range(cnt):
            vec = get_random_unique_vec(vector_size, 0)
            for sub_vec_size in range(13, vector_size):
                if sub_vec_addition(P, Q, vec, sub_vec_size, EC):
                    print("\n something +ve happened...\n")
                    return 123
    return 999


def sub_vec_addition_kth_combo(ele, vec, chose, combo_vec, quota):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()

    cnt = 0
    while cnt < quota:
        total = 0
        for i in range(chose):
            total += vec[combo_vec[i]]
        total %= get_modulus()

        if total == ele % get_modulus():
            print(
                f"\n sum :: {total}\t ele :: {ele}\t cnt :: {cnt}\t vec-size :: {len(vec)}\t sub-vec-size :: {chose}\t pId :: {processor_id}"
            )
            return True

        if not isLastCombination(combo_vec, chose, len(vec)):
            _getNextCombination(combo_vec, len(vec), chose)
        else:
            break
        cnt += 1
    return False


def process_combinations(e, vec, vector_size, sub_vector_size):
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    for sub_vec_size in range(sub_vector_size, vector_size - 1):
        total_combinations = nCr(vector_size, sub_vec_size)
        quota = total_combinations // total
        extra = total_combinations % total

        if processor_id == total - 1:
            quota += extra

        symbol_vec_dim = vector_size
        symbol_vec = list(range(symbol_vec_dim))
        combination_vec = [0] * sub_vec_size
        combination_index = (total_combinations * processor_id) // total
        if processor_id == 0:
            combination_index = 0

        get_kth_combination(symbol_vec, vector_size, sub_vec_size, combination_index, combination_vec)

        if sub_vec_addition_kth_combo(e, vec, sub_vec_size, combination_vec, quota):
            return True
    return False


def hypothesis_A_parallel():
    comm = MPI.COMM_WORLD
    processor_id = comm.Get_rank()
    total = comm.Get_size()

    p = 35184372088639
    vector_size = 39
    max_vector_size = 45
    start_sub_vec_size = 24
    number_of_sub_vectors = 1000

    e = 0
    if processor_id == 0:
        e = random.randrange(p)

    while vector_size <= max_vector_size:
        vec = [0] * vector_size
        sub_vec_size = start_sub_vec_size

        if processor_id == 0:
            position = 1
            if e < (p // 2):
                position = 0

            print(
                f"\n !e :: {e} ({e.bit_length()}b) \t p :: {p} ({p.bit_length()}b) \t vec-size :: {vector_size}\t position :: {position}\t sub-vec-size :: {sub_vec_size}"
            )

            cnt = 0
            while cnt < number_of_sub_vectors:
                vec = get_random_unique_vec(vector_size, e)
                e = comm.bcast(e, root=0)
                vec = comm.bcast(vec, root=0)
                cnt += 1
                if process_combinations(e, vec, vector_size, sub_vec_size):
                    print(f"\n vec-cnt :: {cnt}")
                    comm.Abort(73)
        else:
            cnt = 0
            while cnt < number_of_sub_vectors:
                e = comm.bcast(None, root=0)
                vec = comm.bcast(None, root=0)
                cnt += 1
                if process_combinations(e, vec, vector_size, sub_vec_size):
                    print(f"\n vec-cnt :: {cnt}")
                    comm.Abort(73)

        vector_size += 1


getRandomUniqueVec = get_random_unique_vec
subVec_addition = sub_vec_addition
subVec_addition_kthCombo = sub_vec_addition_kth_combo
processCombinations = process_combinations
hypothesisi_A_parallel = hypothesis_A_parallel
