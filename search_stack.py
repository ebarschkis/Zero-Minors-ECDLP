import numpy as np
from sympy import Matrix

from logging_utils import get_logger
from minors import brute_force_2x2, det_mod


def reconstruct_kernel(non_identity_kernel: np.ndarray, p: int) -> np.ndarray:
    """
    From stored kernel (left block), build (A | I_rev) as in makeKernelFromMatrix.
    If matrix already has >= 2*rows columns, return it modulo p.
    """
    rows, cols = non_identity_kernel.shape
    if cols >= 2 * rows:
        return non_identity_kernel % p
    K = np.zeros((rows, cols * 2), dtype=object)
    K[:, :cols] = non_identity_kernel % p
    for i in range(rows):
        K[i, K.shape[1] - i - 1] = 1
    return K % p


def schur_complement_scan(non_identity_kernel: np.ndarray, p: int, rank: int, world: int) -> tuple:
    """
    Emulates schurComplement_serial: iterate column reductions, compute reduced (A|I) blocks, scan 2x2 minors of the H' block.
    """
    logger = get_logger("schur")
    K_full = reconstruct_kernel(non_identity_kernel, p)
    n = non_identity_kernel.shape[0]

    for offset in range(n):
        # take submatrix removing first `offset` rows/cols on left block
        left = K_full[offset:, :n]
        right = K_full[offset:, n:]
        if left.shape[0] < 2:
            break

        # Gaussian elimination (rref) on left|right over GF(p)
        combined = Matrix(np.concatenate([left, right], axis=1).tolist())
        rref_mat, _ = combined.rref(iszerofunc=lambda x: x % p == 0, modulus=p)
        rref = np.array(rref_mat.tolist(), dtype=object) % p

        # Extract H' block: rows/cols corresponding to remaining columns of left?
        # After elimination, left should be close to identity; take trailing right block rows equal to left rows.
        H_prime = rref[:, left.shape[1]:]
        if H_prime.shape[0] < 2 or H_prime.shape[1] < 2:
            continue

        det_left = int(Matrix(rref[:, : left.shape[1]].tolist()).det() % p)
        if det_left == 0:
            logger.info(f"Schur scan: singular left block after elimination offset={offset}")
            return True, {"offset": offset, "reason": "left_singular"}

        ok, coords = brute_force_2x2(H_prime, p)
        if ok:
            logger.info(f"Schur scan: zero 2x2 minor in H' offset={offset} coords={coords}")
            return True, {"offset": offset, "coords": coords}

    return False, {}


def oblique_elimination_check(non_identity_kernel: np.ndarray, p: int, rank: int, world: int) -> tuple:
    """
    Adaptation of oblique elimination: compute LU over GF(p) and check L_{k,j} and U_{k,j} minors as in C++ get_Ljk_Minor.
    """
    logger = get_logger("oblique")
    A_full = reconstruct_kernel(non_identity_kernel, p)
    n = non_identity_kernel.shape[0]
    A = A_full[:, :n] % p
    M = Matrix(A.tolist())
    try:
        L_sym, U_sym, _ = M.LUdecomposition()
    except Exception as e:
        logger.debug(f"LU failed: {e}")
        return False, {}

    L = np.array(L_sym.tolist(), dtype=object) % p
    U = np.array(U_sym.tolist(), dtype=object) % p
    n = L.shape[0]

    def Ljk_minor(mat: np.ndarray, k: int, j: int) -> np.ndarray:
        rows = list(range(j, j + k + 1))
        cols = list(range(0, k + 1))
        return mat[np.ix_(rows, cols)]

    # Check L_{k,j}
    for j in range(1, n):
        for k in range(1, n - j + 1):
            sub = Ljk_minor(L, k - 1, j - 1)
            if det_mod(sub, p) == 0:
                logger.info(f"Oblique: zero minor in L at k={k-1} j={j-1}")
                return True, {"matrix": "L", "k": k - 1, "j": j - 1}

    # Check U_{k,j} using transpose trick
    U_t = U.T
    for j in range(1, n):
        for k in range(1, n - j + 1):
            sub = Ljk_minor(U_t, k - 1, j - 1).T
            if det_mod(sub, p) == 0:
                logger.info(f"Oblique: zero minor in U at k={k-1} j={j-1}")
                return True, {"matrix": "U", "k": k - 1, "j": j - 1}

    return False, {}
