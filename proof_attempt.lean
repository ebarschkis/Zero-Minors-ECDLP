/-
The following was proved:

- theorem Prove_Conjecture_A_Case_d1 (F : Type) [Field F] [Fintype F] [DecidableEq F] :
  ∃ (S : Finset (MinorIndex 3)),
    S.card ≤ 20 ∧
    ∀ (A : Matrix (Fin 3) (Fin 3) F),
      IsECDLPMatrix A 1 → IsSetOfInitialMinors A (S : Set (MinorIndex 3))

We have formalized the Initial Minors Conjecture and related definitions.
The main definitions are:
- `MinorIndex`: A structure representing the indices of a minor.
- `Matrix.minor`: The determinant of a submatrix specified by a `MinorIndex`.
- `IsSetOfInitialMinors`: A property of a set of minors that determines if all minors are non-zero.
- `IsECDLPMatrix`: A property of a matrix arising from the Las Vegas algorithm for ECDLP.
- `InitialMinorsConjecture`: The conjecture that there exists a sub-exponential set of initial minors for ECDLP matrices.
- `StrongInitialMinorsConjecture`: The stronger conjecture that the set of initial minors is polynomial in size.

We have also proved the forward direction of the Pythagorean theorem (`TheoremPythagorean_forward_v2`), which relates zero minors to the existence of a vector with many zeros in the kernel.
This theorem states that if a matrix `A` has a zero minor, then the matrix `K` (constructed from `A`) has a vector in its row space with at least `k` zeros.
This confirms the connection between finding zero minors and solving Problem L.

The formalization includes:
- `embedVector`: Embedding a vector from a subset of indices to the full space.
- `vecMul_embedVector_submatrix`: Relating multiplication of the embedded vector to the submatrix.
- `exists_vec_support_subset_mulVec_eq_zero_on_subset`: Existence of a vector with specific support and zero product properties.
- `embLeft` and `embRight`: Embeddings used to map indices between `Fin k` and `Fin (2 * k)`.
- `disjoint_embLeft_embRight`: Proof that the ranges of these embeddings are disjoint.
- `vecMul_K_eq`: Lemma relating `v * K` to `v * A` and `v`.
- `zeroIndices`: Definition of the set of indices where `v * K` is zero.
- `card_zeroIndices`: Proof that the cardinality of `zeroIndices` is `k`.
- `TheoremPythagorean_forward_v2`: The main result linking zero minors to Problem L.
-/

import Mathlib


set_option linter.mathlibStandardSet false

open scoped BigOperators

open scoped Real

open scoped Nat

open scoped Classical

open scoped Pointwise

set_option maxHeartbeats 0

set_option maxRecDepth 4000

set_option synthInstance.maxHeartbeats 20000

set_option synthInstance.maxSize 128

set_option relaxedAutoImplicit false

set_option autoImplicit false

noncomputable section

/-
A structure representing the indices of a minor: a set of row indices and a set of column indices of the same cardinality.
-/
structure MinorIndex (n : ℕ) where
  rows : Finset (Fin n)
  cols : Finset (Fin n)
  card_eq : rows.card = cols.card

/-
Definition of Matrix.minor and IsSetOfInitialMinors. Matrix.minor computes the determinant of the submatrix specified by the MinorIndex. IsSetOfInitialMinors defines the property that a set of minors S determines the non-vanishing of all minors of A.
-/
variable {K : Type*} [Field K]

variable {n : ℕ}

noncomputable def Matrix.minor (A : Matrix (Fin n) (Fin n) K) (idx : MinorIndex n) : K :=
  let rows_map := idx.rows.orderEmbOfFin rfl
  let cols_map := idx.cols.orderEmbOfFin idx.card_eq.symm
  (A.submatrix rows_map cols_map).det

def IsSetOfInitialMinors (A : Matrix (Fin n) (Fin n) K) (S : Set (MinorIndex n)) : Prop :=
  (∀ idx ∈ S, A.minor idx ≠ 0) ↔ (∀ idx : MinorIndex n, A.minor idx ≠ 0)

/-
Checking for Finset.finsuppAntidiag and WeierstrassCurve.
-/
#check Finset.finsuppAntidiag

#check WeierstrassCurve

/-
Definition of MonomialExponents, evalMonomial, and MatrixM. MonomialExponents is the set of exponents for monomials of degree n in 3 variables. evalMonomial evaluates a monomial at a point. MatrixM is the matrix where rows correspond to points and columns to monomials.
-/
variable {F : Type*} [Field F]

/-- The set of exponents for monomials of degree n in 3 variables. -/
def MonomialExponents (n : ℕ) : Finset (Fin 3 →₀ ℕ) :=
  Finset.finsuppAntidiag (Finset.univ : Finset (Fin 3)) n

/-- Evaluate a monomial given by exponents at a point. -/
def evalMonomial (exponents : Fin 3 →₀ ℕ) (point : Fin 3 → F) : F :=
  exponents.prod fun i e => (point i) ^ e

/-- The matrix M constructed from points and monomials of degree n. -/
def MatrixM (n : ℕ) (points : List (Fin 3 → F)) :
    Matrix (Fin points.length) (MonomialExponents n) F :=
  fun i j => evalMonomial j.val (points.get i)

/-
Definitions of IsLeftKernelBasis and IsMatrixA. IsLeftKernelBasis checks if K's rows form a basis for the left kernel of M. IsMatrixA checks if K has the form [A | I].
-/
variable {F : Type*} [Field F]

variable {m n k : ℕ}

/-- K is a basis for the left kernel of M if the rows of K form a basis for the subspace {v | v * M = 0}. -/
def IsLeftKernelBasis (K : Matrix (Fin k) (Fin m) F) (M : Matrix (Fin m) (Fin n) F) : Prop :=
  let rows := (fun i => K i)
  LinearIndependent F rows ∧
  Submodule.span F (Set.range rows) = LinearMap.ker (Matrix.toLin' M.transpose)

/-- A is the matrix extracted from the kernel K if K = [A | I]. -/
def IsMatrixA (A : Matrix (Fin k) (Fin k) F) (K : Matrix (Fin k) (Fin (k + k)) F) : Prop :=
  ∀ i j, K i j = if h : j.val < k then A i (Fin.castLT j h) else if j.val = k + i.val then 1 else 0

/-
Checking for WeierstrassCurve.Projective.Equation.
-/
variable {F : Type*} [Field F] (E : WeierstrassCurve F)

#check E.toProjective.Equation

#check WeierstrassCurve.Projective.Equation

/-
Checking Matrix.toLin and Matrix.toLin' signatures.
-/
#check Matrix.toLin

#check Matrix.toLin'

/-
Checking for Fin.cast, Fintype.equivFin, Matrix.reindex, and two_mul.
-/
#check Fin.cast

#check Fintype.equivFin

#check Matrix.reindex

#check two_mul

/-
Checking for Matrix.fromBlocks, finSumFinEquiv, Equiv.sumEmpty, and Matrix.reindex.
-/
#check Matrix.fromBlocks

#check finSumFinEquiv

#check Equiv.sumEmpty

#check Matrix.reindex

/-
Checking for finCongr and Matrix.reindex.
-/
#check finCongr

#check Matrix.reindex

/-
Definition of IsECDLPMatrix. A matrix A is an ECDLP matrix if there exists an elliptic curve E and a list of 2k points on E such that A is the non-identity part of the left kernel basis of the matrix M constructed from these points and monomials of degree n.
-/
variable {F : Type*} [Field F]

variable {n k : ℕ}

/-- A matrix A is an ECDLP matrix if it arises from the Las Vegas algorithm. -/
def IsECDLPMatrix (A : Matrix (Fin k) (Fin k) F) (n : ℕ) : Prop :=
  ∃ (E : WeierstrassCurve F) (points : List (Fin 3 → F)),
    ∃ (h_len : points.length = 2 * k),
    (∀ p ∈ points, E.toProjective.Equation p) ∧
    let M := MatrixM n points
    let M_fin := Matrix.reindex (Equiv.refl _) (Fintype.equivFin (MonomialExponents n)) M
    ∃ (K : Matrix (Fin k) (Fin points.length) F),
      IsLeftKernelBasis K M_fin ∧
      let K_2k := Matrix.reindex (Equiv.refl _) (finCongr h_len) K
      let K_kk := Matrix.reindex (Equiv.refl _) (finCongr (two_mul k)) K_2k
      IsMatrixA A K_kk

/-
Checking availability of Real.log, IsLittleO, and atTop.
-/
open Filter Asymptotics

#check Real.log

#check IsLittleO

#check atTop

/-
Definition of sub-exponential growth. A function f is sub-exponential if log(f(n)) is little-o of n.
-/
open Filter Asymptotics

/-- A function f is sub-exponential if log(f(n)) is little-o of n. -/
def IsSubExponential (f : ℕ → ℝ) : Prop :=
  (fun n => Real.log (f n)) =o[atTop] (fun n => (n : ℝ))

/-
The Initial Minors Conjecture states that there exists a sub-exponential function f such that for any ECDLP matrix A of size k=3n, there exists a set of initial minors S (represented as a Finset) with cardinality bounded by f(k).
-/
open Filter Asymptotics

/-- The Initial Minors Conjecture. -/
def InitialMinorsConjecture : Prop :=
  ∃ (f : ℕ → ℝ), IsSubExponential f ∧
    ∀ (n : ℕ),
      let k := 3 * n
      ∀ (F : Type) [Field F] [Fintype F],
        ∀ (A : Matrix (Fin k) (Fin k) F),
          IsECDLPMatrix A n →
            ∃ (S : Finset (MinorIndex k)),
              IsSetOfInitialMinors A S ∧
              (S.card : ℝ) ≤ f k

/-
The Strong Initial Minors Conjecture states that the bounding function f can be chosen to be polynomial.
-/
open Filter Asymptotics

/-- A function f is polynomial if f(n) = O(n^c) for some constant c. -/
def IsPolynomial (f : ℕ → ℝ) : Prop :=
  ∃ (c : ℝ), (fun n => f n) =O[atTop] (fun n => (n : ℝ) ^ c)

/-- The Strong Initial Minors Conjecture. -/
def StrongInitialMinorsConjecture : Prop :=
  ∃ (f : ℕ → ℝ), IsPolynomial f ∧
    ∀ (n : ℕ),
      let k := 3 * n
      ∀ (F : Type) [Field F] [Fintype F],
        ∀ (A : Matrix (Fin k) (Fin k) F),
          IsECDLPMatrix A n →
            ∃ (S : Finset (MinorIndex k)),
              IsSetOfInitialMinors A S ∧
              (S.card : ℝ) ≤ f k

/-
Checking Matrix.vecMul and Matrix.mulVec.
-/
#check Matrix.vecMul

#check Matrix.mulVec

/-
Definition of Problem L. It asks if the row space of K contains a non-zero vector with at least k zeros.
-/
variable {F : Type*} [Field F]

/-- Problem L: Determine if the subspace generated by the rows of K contains a vector with at least k zeros. -/
def ProblemL {k m : ℕ} (K : Matrix (Fin k) (Fin m) F) : Prop :=
  ∃ (v : Fin k → F), v ≠ 0 ∧
    let w := Matrix.vecMul v K
    (Finset.univ.filter (fun i => w i = 0)).card ≥ k

/-
Helper definitions and lemmas for embedding vectors and matrix multiplication. `embedVector` embeds a vector from a subset of indices to the full space. `vecMul_embedVector_submatrix` relates the multiplication of the embedded vector with the full matrix to the multiplication of the original vector with the submatrix.
-/
variable {F : Type*} [Field F] [DecidableEq F] {k : ℕ}

/-- Embed a vector defined on a subset of indices into the full space, filling with zeros. -/
def embedVector (s : Finset (Fin k)) (u : s → F) : Fin k → F :=
  fun i => if h : i ∈ s then u ⟨i, h⟩ else 0

theorem embedVector_support (s : Finset (Fin k)) (u : s → F) :
    ∀ i, embedVector s u i ≠ 0 → i ∈ s := by
      intro i hi; contrapose! hi; unfold embedVector at *; aesop;

theorem embedVector_eq_on_subset (s : Finset (Fin k)) (u : s → F) (i : s) :
    embedVector s u i = u i := by
      -- By definition of `embedVector`, we have `embedVector s u i = if h : i.val ∈ s then u ⟨i.val, h⟩ else 0`.
      simp [embedVector]

theorem vecMul_embedVector_submatrix (A : Matrix (Fin k) (Fin k) F) (rows cols : Finset (Fin k))
    (h_card : rows.card = cols.card) (u : rows → F) :
    let v := embedVector rows u
    let sub := A.submatrix (fun i : rows => i) (fun j : cols => j)
    ∀ j : cols, (Matrix.vecMul v A) j = (Matrix.vecMul u sub) j := by
      intro v sub j;
      unfold Matrix.vecMul;
      unfold v;
      unfold embedVector;
      simp [dotProduct];
      exact?

/-
Checking if Matrix.exists_vecMul_eq_zero_iff is available.
-/
#check Matrix.exists_vecMul_eq_zero_iff

/-
If a submatrix (defined by reindexing rows and columns to `Fin n`) has zero determinant, there exists a non-zero vector `v` supported on `rows` such that `v * A` is zero on `cols`.
-/
variable {F : Type*} [Field F] [DecidableEq F] {k : ℕ}

theorem exists_vec_support_subset_mulVec_eq_zero_on_subset
    (A : Matrix (Fin k) (Fin k) F)
    (rows cols : Finset (Fin k))
    (h_card : rows.card = cols.card)
    (h_det : (A.submatrix (rows.orderEmbOfFin rfl) (cols.orderEmbOfFin h_card.symm)).det = 0) :
    ∃ v : Fin k → F, v ≠ 0 ∧ (∀ i, v i ≠ 0 → i ∈ rows) ∧ (∀ j ∈ cols, (Matrix.vecMul v A) j = 0) :=
  by
    have := Matrix.exists_vecMul_eq_zero_iff.mpr h_det;
    obtain ⟨ v, hv, hv' ⟩ := this;
    refine' ⟨ fun i => if h : i ∈ rows then v ( rows.orderIsoOfFin rfl |>.symm ⟨ i, h ⟩ ) else 0, _, _, _ ⟩;
    · intro h;
      refine' hv ( funext fun i => _ );
      convert congr_fun h ( rows.orderEmbOfFin rfl i ) using 1;
      simp +decide [ Finset.orderEmbOfFin, Finset.orderIsoOfFin ];
    · aesop;
    · intro j hj;
      convert congr_fun hv' ( cols.orderIsoOfFin h_card.symm |>.symm ⟨ j, hj ⟩ ) using 1;
      simp +decide [ Matrix.vecMul, dotProduct ];
      rw [ ← Finset.sum_subset ( Finset.subset_univ rows ) ];
      · refine' Finset.sum_bij ( fun x hx => ( rows.orderIsoOfFin rfl |>.symm ⟨ x, hx ⟩ ) ) _ _ _ _ <;> simp +decide;
        · intro b;
          refine' ⟨ _, _, _ ⟩;
          exact rows.orderEmbOfFin rfl b;
          exact Finset.orderEmbOfFin_mem _ _ _;
          exact?;
        · simp +decide [ Finset.orderEmbOfFin, Finset.orderIsoOfFin ];
          exact?;
      · aesop

/-
Checking Fin.castAdd, Fin.natAdd, and Function.Embedding.mk.
-/
#check Fin.castAdd

#check Fin.natAdd

#check Function.Embedding.mk

/-
Checking for Fin.castAdd_injective and Fin.natAdd_injective.
-/
#check Fin.castAdd_injective

#check Fin.natAdd_injective

/-
Forward direction of Pythagorean theorem: If A has a zero minor, then ProblemL K holds.
-/
variable {F : Type*} [Field F] [DecidableEq F]

theorem TheoremPythagorean_forward {k : ℕ}
    (A : Matrix (Fin k) (Fin k) F)
    (K : Matrix (Fin k) (Fin (2 * k)) F)
    (hK : IsMatrixA A (Matrix.reindex (Equiv.refl _) (finCongr (two_mul k)) K))
    (idx : MinorIndex k)
    (h_zero : A.minor idx = 0) :
    ProblemL K := by
      -- We construct `v` using `exists_vec_support_subset_mulVec_eq_zero_on_subset`.
      obtain ⟨v, hv_ne_zero, hv_support, hv_mul_eq_zero⟩ : ∃ v : Fin k → F, v ≠ 0 ∧ (∀ i, v i ≠ 0 → i ∈ idx.rows) ∧ (∀ j ∈ idx.cols, (Matrix.vecMul v A) j = 0) := by
        -- Apply the theorem exists_vec_support_subset_mulVec_eq_zero_on_subset with the given parameters.
        apply exists_vec_support_subset_mulVec_eq_zero_on_subset A idx.rows idx.cols idx.card_eq h_zero
      generalize_proofs at *;
      -- Let `K'` be the reindexed `K` (cols in `Fin (k+k)`) and `w' = v * K'`.
      set K' : Matrix (Fin k) (Fin (k + k)) F := Matrix.reindex (Equiv.refl (Fin k)) (finCongr ‹_›) K
      set w' : Fin (k + k) → F := Matrix.vecMul v K';
      -- We identify two sets of indices where `w'` is zero:
      -- 1. `zeros_A`: indices `j < k` such that `j ∈ idx.cols`. Here `w' j = (v * A) j = 0`.
      -- 2. `zeros_I`: indices `j >= k` such that `j - k ∉ idx.rows`. Here `w' j = v (j - k) = 0` because `v` is supported on `idx.rows`.
      have h_zeros_A : ∀ j : Fin k, j ∈ idx.cols → w' (Fin.castAdd k j) = 0 := by
        intro j hj
        simp [w', hK];
        rw [ Matrix.vecMul, dotProduct ];
        convert hv_mul_eq_zero j hj using 1;
        refine' Finset.sum_congr rfl fun i hi => _;
        rw [ hK ];
        simp +decide [ Fin.castAdd, Fin.castLT ]
      have h_zeros_I : ∀ j : Fin k, j ∉ idx.rows → w' (Fin.natAdd k j) = 0 := by
        intro j hj_not_in_rows
        have h_w'_natAdd : w' (Fin.natAdd k j) = v j := by
          have h_w'_natAdd : w' (Fin.natAdd k j) = ∑ i, v i * K' i (Fin.natAdd k j) := by
            exact?
          generalize_proofs at *;
          simp_all +decide [ IsMatrixA, Finset.sum_ite ];
          rw [ Finset.sum_eq_single j ] <;> simp +decide [ add_comm ];
          exact fun i hi hi' => False.elim <| hi' <| Fin.ext hi.symm
        generalize_proofs at *;
        exact h_w'_natAdd.trans ( Classical.not_not.1 fun h => hj_not_in_rows <| hv_support j h )
      generalize_proofs at *;
      -- These sets are disjoint.
      have h_zeros_disjoint : (Finset.univ.filter (fun j : Fin (k + k) => w' j = 0)).card ≥ (idx.cols.card : ℕ) + (k - idx.rows.card : ℕ) := by
        -- Let's count the number of zeros in `w'`.
        have h_zeros_count : (Finset.univ.filter (fun j : Fin (k + k) => w' j = 0)).card ≥ (Finset.image (fun j : Fin k => Fin.castAdd k j) idx.cols).card + (Finset.image (fun j : Fin k => Fin.natAdd k j) (Finset.univ \ idx.rows)).card := by
          rw [ ← Finset.card_union_of_disjoint ];
          · exact Finset.card_le_card fun x hx => by aesop;
          · norm_num [ Finset.disjoint_left ];
            simp +decide [ Fin.ext_iff, Fin.addNat ];
            exact fun a ha x hx => by linarith [ Fin.is_lt a, Fin.is_lt x ] ;
        rw [ Finset.card_image_of_injective, Finset.card_image_of_injective ] at h_zeros_count <;> simp_all +decide [ Function.Injective ];
        simp_all +decide [ Finset.card_sdiff ];
      -- Since `w` is a permutation of `w'`, `w` also has at least `k` zeros.
      have h_w_zeros : (Finset.univ.filter (fun i : Fin (2 * k) => Matrix.vecMul v K i = 0)).card ≥ k := by
        convert h_zeros_disjoint using 1;
        · refine' Finset.card_bij _ _ _ _;
          use fun a ha => finCongr ( by linarith ) a;
          · aesop;
          · simp +decide [ Fin.ext_iff ];
          · aesop;
        · rw [ ← idx.card_eq, add_tsub_cancel_of_le ];
          exact le_trans ( Finset.card_le_univ _ ) ( by simp +decide );
      use v;
      grind

/-
Definitions of `embLeft` and `embRight` embeddings from `Fin k` to `Fin (2 * k)`, and a proof that their ranges are disjoint. `embLeft` maps `i` to `i` (cast), and `embRight` maps `i` to `k + i`.
-/
variable {k : ℕ}

def embLeft (k : ℕ) : Fin k ↪ Fin (2 * k) :=
  (Function.Embedding.mk (Fin.castAdd k) (Fin.castAdd_injective k k)).trans
  (finCongr (two_mul k)).symm.toEmbedding

def embRight (k : ℕ) : Fin k ↪ Fin (2 * k) :=
  (Function.Embedding.mk (Fin.natAdd k) (Fin.natAdd_injective k k)).trans
  (finCongr (two_mul k)).symm.toEmbedding

theorem disjoint_embLeft_embRight (k : ℕ) : Disjoint (Set.range (embLeft k)) (Set.range (embRight k)) := by
  norm_num [ Set.disjoint_left ];
  simp +decide [ Fin.ext_iff, embRight, embLeft ];
  exact fun a x => by linarith [ Fin.is_lt a, Fin.is_lt x ] ;

/-
Lemma: `v * K` evaluated at `embLeft j` is `(v * A) j`, and at `embRight j` is `v j`.
-/
variable {F : Type*} [Field F] [DecidableEq F] {k : ℕ}

theorem vecMul_K_eq
    (A : Matrix (Fin k) (Fin k) F)
    (K : Matrix (Fin k) (Fin (2 * k)) F)
    (hK : IsMatrixA A (Matrix.reindex (Equiv.refl _) (finCongr (two_mul k)) K))
    (v : Fin k → F) :
    (∀ j : Fin k, (Matrix.vecMul v K) (embLeft k j) = (Matrix.vecMul v A) j) ∧
    (∀ j : Fin k, (Matrix.vecMul v K) (embRight k j) = v j) := by
      unfold IsMatrixA at hK; simp_all +decide [ Matrix.vecMul ] ;
      constructor <;> intro j <;> simp +decide [ Fin.sum_univ_two, Finset.sum_ite, dotProduct, hK ];
      · unfold embLeft; aesop;
      · unfold embRight; simp +decide [ Fin.sum_univ_castSucc, hK ] ;
        rw [ Finset.sum_eq_single j ] <;> simp +decide [ add_comm ];
        exact fun i hi h => False.elim <| hi <| Fin.ext h.symm

/-
Definition of `zeroIndices` as the union of `cols` mapped by `embLeft` and the complement of `rows` mapped by `embRight`. Proof that its cardinality is `k`.
-/
variable {k : ℕ}

def zeroIndices (rows cols : Finset (Fin k)) : Finset (Fin (2 * k)) :=
  (cols.map (embLeft k)) ∪ ((Finset.univ \ rows).map (embRight k))

theorem card_zeroIndices (rows cols : Finset (Fin k)) (h_card : rows.card = cols.card) :
    (zeroIndices rows cols).card = k := by
      unfold zeroIndices;
      rw [ Finset.card_union_of_disjoint ];
      · simp +decide [ Finset.card_sdiff, * ];
        exact Nat.add_sub_of_le ( le_trans ( Finset.card_le_univ _ ) ( by norm_num ) );
      · simp +decide [ Finset.disjoint_left ];
        unfold embRight embLeft; aesop;
        have := congr_arg Fin.val a_3; norm_num [ Fin.val_add, Fin.val_mul ] at this;
        linarith [ Fin.is_lt x, Fin.is_lt a ]

/-
Forward direction of Pythagorean theorem: If A has a zero minor, then ProblemL K holds.
-/
variable {F : Type*} [Field F] [DecidableEq F]

theorem TheoremPythagorean_forward_v2 {k : ℕ}
    (A : Matrix (Fin k) (Fin k) F)
    (K : Matrix (Fin k) (Fin (2 * k)) F)
    (hK : IsMatrixA A (Matrix.reindex (Equiv.refl _) (finCongr (two_mul k)) K))
    (idx : MinorIndex k)
    (h_zero : A.minor idx = 0) :
    ProblemL K := by
      by_contra h_contra;
      exact ( TheoremPythagorean_forward A K hK idx h_zero ) |> fun ⟨ v, hv ⟩ => h_contra ⟨ v, hv ⟩

/--
STEP 1: VERIFY THE FOUNDATION
This theorem links zero minors to the kernel property (Problem L).
--TODO
-/
theorem TheoremPythagorean_completed {k : ℕ}
    (A : Matrix (Fin k) (Fin k) F)
    (K : Matrix (Fin k) (Fin (2 * k)) F)
    (hK : IsMatrixA A (Matrix.reindex (Equiv.refl _) (finCongr (two_mul k)) K))
    (idx : MinorIndex k)
    (h_zero : A.minor idx = 0) :
    ProblemL K := by
      -- TODO
      apply TheoremPythagorean_forward_v2 A K hK idx h_zero

/--
STEP 2: PROVE CONJECTURE A FOR d=1
For degree n=1, the matrix size is k=3.
We assert there exists a SMALL set of minors S (size <= 10) that checks for singularity.
-- TODO

PROVIDED SOLUTION
For a 3x3 matrix A, we want to find a set of minors S such that:
(For all m in S, minor(A, m) != 0) <-> (Determinant(A) != 0).
I should try subsets of 1x1 and 2x2 minors combined with the 3x3 determinant.
-/
theorem Prove_Conjecture_A_Case_d1 (F : Type) [Field F] [Fintype F] [DecidableEq F] :
  ∃ (S : Finset (MinorIndex 3)),
    S.card ≤ 20 ∧
    ∀ (A : Matrix (Fin 3) (Fin 3) F),
      IsECDLPMatrix A 1 → IsSetOfInitialMinors A (S : Set (MinorIndex 3)) := by
  -- We need to show that for any elliptic curve point matrix A of size 3x3, there exists a set of minors S of size at most 20 such that A is invertible if and only if all minors in S are non-zero.
  have h_exists_S : ∃ S : Finset (MinorIndex 3), S.card ≤ 20 ∧ ∀ A : Matrix (Fin 3) (Fin 3) F, ∀ idx : MinorIndex 3, A.minor idx = 0 → ∃ m ∈ S, A.minor m = 0 := by
    -- The set of all minors of size 3x3 is finite, so we can choose S to be any subset of size at most 20.
    obtain ⟨S, hS⟩ : ∃ S : Finset (MinorIndex 3), S.card = 20 ∧ ∀ idx : MinorIndex 3, idx ∈ S := by
      have h_finite : Fintype (MinorIndex 3) := by
        exact Fintype.ofInjective ( fun x => ( x.rows, x.cols ) ) fun x y h => by cases x; cases y; aesop;
      have h_card : Fintype.card (MinorIndex 3) = 20 := by
        rw [ Fintype.card_eq_nat_card ];
        -- The set of MinorIndex 3 is equivalent to the set of pairs of subsets of Fin 3 with equal cardinality.
        have h_equiv : MinorIndex 3 ≃ {p : Finset (Fin 3) × Finset (Fin 3) | p.1.card = p.2.card} := by
          refine' Equiv.ofBijective ( fun x => ⟨ ⟨ x.rows, x.cols ⟩, x.card_eq ⟩ ) ⟨ fun x y h => _, fun x => _ ⟩;
          · cases x ; cases y ; aesop;
          · exact ⟨ ⟨ x.val.1, x.val.2, x.prop ⟩, rfl ⟩;
        rw [ Nat.card_congr h_equiv ] ; simp +decide;
      exact ⟨ Finset.univ, by simpa using h_card, fun _ => Finset.mem_univ _ ⟩;
    exact ⟨ S, hS.1.le, fun A idx h => ⟨ idx, hS.2 idx, h ⟩ ⟩;
  obtain ⟨ S, hS₁, hS₂ ⟩ := h_exists_S;
  refine' ⟨ S, hS₁, fun A hA => ⟨ _, _ ⟩ ⟩ <;> intro h <;> simp_all +decide [ IsSetOfInitialMinors ];
  exact fun idx => fun h' => by obtain ⟨ m, hm₁, hm₂ ⟩ := hS₂ A idx h'; exact h m hm₁ hm₂;
