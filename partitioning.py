import math
from typing import List, Dict

from combinatorics import nCr


def compute_partition_data_2x2(r: int, total_procs: int) -> List[Dict[str, int]]:
    """
    Port of computePartitionData_2x2: distribute nCr(r,2) combos across processors.
    """
    combos = nCr(r, 2)
    per = combos // total_procs
    extra = combos % total_procs
    pD: List[Dict[str, int]] = []
    # initialize first partition
    pD.append({"i_start": 0, "j_start": 1, "quota": per})
    idx = 0
    cnt = 0
    for i in range(r):
        for j in range(i + 1, r):
            cnt += 1
            if cnt == per and idx < total_procs - 1:
                idx += 1
                quota = per
                if idx == total_procs - 1:
                    quota += extra
                pD.append({"i_start": i, "j_start": j, "quota": quota})
                cnt = 0
    # ensure length
    while len(pD) < total_procs:
        pD.append({"i_start": r, "j_start": r, "quota": per})
    # add extra combos to last
    pD[-1]["quota"] = per + extra
    return pD


def partition_linear(total: int, rank: int, world: int) -> Dict[str, int]:
    """
    Partition a linear index space [0,total) into contiguous chunks.
    """
    start = (total * rank) // world
    end = (total * (rank + 1)) // world
    return {"start": start, "end": end}
