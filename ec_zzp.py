import secrets

import numpy as np

from ec_zzp_point import EC_ZZp_Point


def _mod_inv(x: int, p: int) -> int:
    return pow(int(x % p), -1, p)


class EC_ZZp:
    def __init__(self, p, a4=None, a6=None, ordP=None):
        self.p = int(p)
        if a4 is None or a6 is None:
            self.generateRandomCurve()
        else:
            self.a4 = int(a4) % self.p
            self.a6 = int(a6) % self.p
        self.discriminant = 0

    def address(self):
        return self

    def generateRandomCurve(self):
        self.a4 = secrets.randbelow(self.p)
        self.a6 = secrets.randbelow(self.p)

    def generateRandomPoint(self):
        while True:
            P1 = EC_ZZp_Point()
            P1.x = secrets.randbelow(self.p)
            P1.y = secrets.randbelow(self.p)
            if self.isPointValid(P1):
                return P1

    def printCurve(self):
        print(
            f"Elliptic Curve Defined by y^2 = x^3 + {self.a4}*x + {self.a6} over Finite Field of size {self.p}"
        )

    def isPointValid(self, P: EC_ZZp_Point):
        if P.x == 0 and P.y == 1 and P.z == 0:
            return True
        ans = (P.x * P.x * P.x + self.a4 * P.x + self.a6 - (P.y * P.y)) % self.p
        return ans == 0

    def pointAddition_Doubling(self, P: EC_ZZp_Point, Q: EC_ZZp_Point, ans: EC_ZZp_Point):
        if P.x == 0 and P.y == 1 and P.z == 0:
            ans.x, ans.y, ans.z = Q.x, Q.y, Q.z
            return
        if Q.x == 0 and Q.y == 1 and Q.z == 0:
            ans.x, ans.y, ans.z = P.x, P.y, P.z
            return

        if (P.x == Q.x) and (P.y == Q.y):
            A = (self.a4 + 3 * (P.x * P.x)) % self.p
            B = P.y % self.p
            C = (P.x * P.y * B) % self.p
            D = (A * A - 8 * C) % self.p

            ans.x = (2 * B * D) % self.p
            ans.y = (A * ((4 * C - D) % self.p) - ((8 * P.y * P.y) % self.p) * (B * B)) % self.p
            ans.z = (8 * (B * B * B)) % self.p

            inv_z = _mod_inv(ans.z, self.p)
            ans.x = (ans.x * inv_z) % self.p
            ans.y = (ans.y * inv_z) % self.p
            ans.z = 1
            return

        if P.x == Q.x:
            ans.x = 0
            ans.y = 1
            ans.z = 0
            return

        A = (Q.y - P.y) % self.p
        B = (Q.x - P.x) % self.p
        C = (A * A - (B * B * B) - 2 * (B * B) * P.x) % self.p

        ans.x = (B * C) % self.p
        ans.y = (A * ((B * B * P.x) - C) - ((B * B * B) * P.y)) % self.p
        ans.z = (B * B * B) % self.p

        inv_z = _mod_inv(ans.z, self.p)
        ans.x = (ans.x * inv_z) % self.p
        ans.y = (ans.y * inv_z) % self.p
        ans.z = 1

    def scalarMultiplicationDA(self, P: EC_ZZp_Point, e: int, Q: EC_ZZp_Point):
        num_bits = int(e).bit_length()
        Q.x, Q.y, Q.z = 0, 1, 0

        for i in range(num_bits - 1, -1, -1):
            b = (int(e) >> i) & 1
            tmpP = EC_ZZp_Point()
            self.pointAddition_Doubling(Q, Q, tmpP)
            Q.x, Q.y, Q.z = tmpP.x, tmpP.y, tmpP.z
            if b:
                tmpP1 = EC_ZZp_Point()
                self.pointAddition_Doubling(Q, P, tmpP1)
                Q.x, Q.y, Q.z = tmpP1.x, tmpP1.y, tmpP1.z

    def scalarMultiplication_Basic(self, P: EC_ZZp_Point, e: int, ans: EC_ZZp_Point):
        cnt = 2
        self.pointAddition_Doubling(P, P, ans)
        while cnt < e:
            tmp_p = EC_ZZp_Point()
            self.pointAddition_Doubling(P, ans, tmp_p)
            ans.x, ans.y, ans.z = tmp_p.x, tmp_p.y, tmp_p.z
            cnt += 1

    def pointNegation(self, P: EC_ZZp_Point, Q: EC_ZZp_Point):
        Q.x = P.x
        Q.y = (-P.y) % self.p
        Q.z = 1

    def order(self, P: EC_ZZp_Point):
        cnt = 1
        P_tmp = EC_ZZp_Point()
        P_tmp.x, P_tmp.y, P_tmp.z = P.x, P.y, P.z

        while True:
            if P.x == 0 and P.y == 1 and P.z == 0:
                return cnt

            P2 = EC_ZZp_Point()
            self.pointAddition_Doubling(P_tmp, P_tmp, P2)
            cnt += 1

            P_tmp.x, P_tmp.y, P_tmp.z = P2.x, P2.y, P2.z
            print(f"\n cnt :: {cnt}")
            P2.printPoint("\tP2 :: ")
            P_tmp.printPoint("\t\tP_tmp :: ")

            if cnt > 2000:
                break
        return 0

    def generateMatrix(self, M, P, Q, k_random_nums, t_random_nums, PQ_random_numbers, weighted_vector_arr):
        for i in range(k_random_nums):
            P1 = EC_ZZp_Point()
            self.scalarMultiplicationDA(P, int(PQ_random_numbers[i]), P1)
            if P1.x == Q.x and P1.y == Q.y:
                return 1
            for j in range(M.shape[1]):
                M[i, j] = (pow(P1.x, weighted_vector_arr[j][0], self.p) * pow(P1.y, weighted_vector_arr[j][1], self.p)) % self.p

        for i in range(k_random_nums, k_random_nums + t_random_nums):
            P1 = EC_ZZp_Point()
            self.pointNegation(Q, P1)
            P2 = EC_ZZp_Point()
            self.scalarMultiplicationDA(P1, int(PQ_random_numbers[i]), P2)
            if P2.x == Q.x and P2.y == Q.y:
                return 1
            for j in range(M.shape[1]):
                M[i, j] = (pow(P2.x, weighted_vector_arr[j][0], self.p) * pow(P2.y, weighted_vector_arr[j][1], self.p)) % self.p
        return 0

    def generateMatrix2(self, M, P, Q, k_random_nums, t_random_nums, PQ_random_numbers, weighted_vector_arr):
        return self.generateMatrix(M, P, Q, k_random_nums, t_random_nums, PQ_random_numbers, weighted_vector_arr)

    def generateMatrix_Random(self, M, P, Q, k_random_nums, t_random_nums, PQ_random_numbers, weighted_vector_arr):
        return self.generateMatrix(M, P, Q, k_random_nums, t_random_nums, PQ_random_numbers, weighted_vector_arr)

    def lasVegasECDLP_1(self, P, Q, ordP):
        from apm_port import Curve, Point, las_vegas

        curve = Curve(self.p, self.a4, self.a6)
        Pp = Point(P.x, P.y, False)
        Qp = Point(Q.x, Q.y, False)
        return las_vegas(curve, Pp, Qp, int(ordP), int(ordP).bit_length(), 1)

    def lasVegasECDLP_2(self, P, Q, ordP):
        return self.lasVegasECDLP_1(P, Q, ordP)

    def lasVegasECDLP_3(self, P, Q, ordP):
        return self.lasVegasECDLP_1(P, Q, ordP)

    def generateKernels_ZZp(self, P, Q, ordP, offset):
        from ec_las_vegas_impl import genetateKernels

        n_bits = int(ordP).bit_length()
        genetateKernels(P, Q, ordP, n_bits, offset, self)
        return 0


def printMatrix1(mat):
    print(mat)


EC_ZZp = EC_ZZp
