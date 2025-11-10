#!/usr/bin/env python3
# ===========================================================
# Isolator growth study (parameterized target ζ(R)):
# - For a given odd TARGET_R >= 9, enforce prior four odd zetas:
#   constraints = {R-2, R-4, R-6, R-8} (filtered to >=3)
# - Records S, H, S-H, ln den(D_R), ||u||_∞, support, patterns
# - Saves CSV: zeta{R}_sweep.csv
# ===========================================================
import sys
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)   # 0 = unlimited

from itertools import product
from fractions import Fraction as Fr
from math import gcd
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, csv
import numpy as np
import mpmath as mp

# ---------------- CONFIG ----------------
TARGET_R = int(sys.argv[1]) if len(sys.argv) > 1 else 13
if TARGET_R % 2 == 0 or TARGET_R < 9:
    raise SystemExit("TARGET_R must be an odd integer ≥ 9")

A = 5
ALPHA_RANGE = (-4, -3, -2, -1)
N_VALUES = list(range(1, 150, 2))
FOCUS_A = lambda n: [max(1.0, n/np.e)]
MAX_NULLSPACE_DIM = 80
MAX_WORKERS = max(1, (os.cpu_count() or 8) - 1)
MP_DPS = 230  # working precision for logs/zeta
CSV_PATH = f"zeta{TARGET_R}_sweep.csv"
# ----------------------------------------

# ----- target & constraints -----
def prior_four(r):
    """Return the 4 previous odd indices for zeta, filtered to >=3."""
    L = [r-2, r-4, r-6, r-8]
    L = [x for x in L if x >= 3 and x % 2 == 1]
    # If fewer than 4 exist (e.g., r=9), pad with the largest available smaller odds down to 3
    need = 4 - len(L)
    k = r-10
    while need > 0 and k >= 3:
        if k % 2 == 1 and k not in L:
            L.append(k); need -= 1
        k -= 2
    return sorted(L)  # ascending

CONSTRAINT_RS = tuple(prior_four(TARGET_R))
R_SET = tuple(sorted(set(CONSTRAINT_RS + (TARGET_R,))))

# ===== exact-series helpers (Fraction arithmetic) =====
def _poly_mul_frac(p, q):
    r = [Fr(0) for _ in range(len(p) + len(q) - 1)]
    for i, a in enumerate(p):
        if a == 0: continue
        for j, b in enumerate(q):
            if b == 0: continue
            r[i+j] += a*b
    return r

def _rising_poly_frac(b, n):
    poly = [Fr(1)]
    for j in range(n):
        poly = _poly_mul_frac(poly, [Fr(b + j), Fr(1)])
    return poly

def _series_inv_frac(p, deg):
    inv0 = Fr(1)/p[0]
    q = [inv0]
    for k in range(1, deg+1):
        s = Fr(0)
        for i in range(1, k+1):
            if i < len(p) and (k-i) < len(q):
                s += p[i]*q[k-i]
        q.append(-inv0*s)
    return q

def _series_mul_frac(p, q, deg):
    r = [Fr(0) for _ in range(deg+1)]
    for i, a in enumerate(p):
        if i > deg: break
        if a == 0: continue
        for j, b in enumerate(q):
            if i+j > deg: break
            if b == 0: continue
            r[i+j] += a*b
    return r

def _v_series_frac(p):
    for i, c in enumerate(p):
        if c != 0: return i
    return len(p)

def Dr_exact_for_column(n, a, alphas, A=A, rset=R_SET):
    """
    Build the exact coefficient map {r: D_r} for r in rset.
    Structure matches earlier derivation; symmetric factor ((3^r-3)/3^r).
    """
    # P = (x)_n as rising poly; P5 = P^5
    P  = _rising_poly_frac(0, n)
    P5 = P
    for _ in range(4):
        P5 = _poly_mul_frac(P5, P)

    # numerator = product_j (α_j - a + t)_n, then shift by 1: [0] + Num
    Num = [Fr(1)]
    for α in alphas:
        Num = _poly_mul_frac(Num, _rising_poly_frac(α - a, n))
    Num = [Fr(0)] + Num

    # divide Num/P5 in series (after stripping valuation)
    deg = 2*A + 2 - 3
    v = _v_series_frac(P5)
    Den = P5[v:]
    Num2 = Num[v:]
    invD = _series_inv_frac(Den, deg)
    H = _series_mul_frac(Num2, invD, deg)

    def c_r(r):
        m = (2*A + 2) - r
        return H[m] if 0 <= m <= deg else Fr(0)

    def sym_fac(r):  # ((3^r - 3)/3^r), harmless for odd r >= 3
        return Fr((3**r) - 3, 3**r)

    return {r: c_r(r)*sym_fac(r) for r in rset}

# top-level worker (pickleable)
def worker_compute(j, col_spec):
    return j, Dr_exact_for_column(*col_spec)

# ---------- exact RREF over Q ----------
def rref_q(M):
    R = [[Fr(x) for x in row] for row in M]
    rows = len(R); cols = len(R[0])
    piv=[]; r=0
    for c in range(cols):
        pr=None
        for i in range(r, rows):
            if R[i][c] != 0:
                pr=i; break
        if pr is None: continue
        R[r], R[pr] = R[pr], R[r]
        inv = Fr(1)/R[r][c]
        R[r] = [inv*x for x in R[r]]
        for i in range(rows):
            if i == r: continue
            if R[i][c] != 0:
                fac = R[i][c]
                R[i] = [R[i][j] - fac*R[r][j] for j in range(cols)]
        piv.append((r, c))
        r += 1
        if r == rows: break
    return R, piv

def nullspace_basis_q(M):
    R, piv = rref_q(M)
    rows = len(R); cols = len(R[0])
    pivcols = {c for _, c in piv}
    free = [j for j in range(cols) if j not in pivcols]
    out = []
    for f in free:
        u = [Fr(0)]*cols; u[f] = Fr(1)
        for r, c in reversed(piv):
            s = Fr(0)
            for j in range(c+1, cols):
                s += R[r][j]*u[j]
            u[c] = -s
        den = 1
        for x in u:
            den = (den*x.denominator)//gcd(den, x.denominator)
        v = [int(x*den) for x in u]
        g = 0
        for t in v: g = gcd(g, abs(t))
        if g > 1: v = [t//g for t in v]
        out.append(v)
    return out

# ---------- numeric pieces ----------
def zeta_vals(rset):
    mp.mp.dps = MP_DPS
    return {r: mp.zeta(r) for r in rset}

def compute_S_H(u, Dr_cols, Z, r_target):
    # Compute D_target and L in exact / high-precision
    D = Fr(0)
    L = mp.mpf('0')
    for j, c in enumerate(u):
        if not c: continue
        Dr = Dr_cols[j]
        # exact target contribution
        D += Fr(c) * Dr[r_target]
        # linear form
        s = mp.mpf('0')
        for r in Z.keys():
            coeff = Dr[r]
            if coeff:
                s += (mp.mpf(coeff.numerator) / mp.mpf(coeff.denominator)) * Z[r]
        L += mp.mpf(c) * s

    # S = ln |numerator(D)| (as in earlier scripts)
    num = abs(D.numerator)
    S = mp.log(mp.mpf(num)) if num != 0 else mp.mpf('0')
    H = mp.log(abs(L)) if L != 0 else mp.mpf('0')
    return float(S), float(H), float(S - H), D, L

# ---------------- main sweep ----------------
def main():
    print(f"\n=== ζ({TARGET_R}) growth sweep ===\n")
    # CSV header
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_r","n","a","S","H","S_minus_H","ln_den_Dtarget","ln_abs_L","u_max","support_size","support_indices","constraints"])

    Z = zeta_vals(R_SET)  # includes constraints + target

    for n in N_VALUES:
        for a in FOCUS_A(n):
            # canonical 5-pattern family + small neighborhood (same as before)
            base = {
                (-4,-4,-4,-4,-4),
                (-4,-4,-4,-4,-3),
                (-4,-4,-4,-4,-2),
                (-4,-4,-4,-4,-1),
                (-4,-4,-4,-3,-3),
            }
            neighborhood = set()
            for x in range(-4, -1):      # -4,-3,-2
                for y in range(-4, 0):   # -4,-3,-2,-1
                    neighborhood.add((-4,-4,-4,x,y))
                    neighborhood.add((-4,-4,x,y,-3))
            patterns = sorted(base | neighborhood)

            cols = [(n, a, p) for p in patterns]
            Dr_cols = [None]*len(cols)

            # parallel compute Dr for all columns with the current R_SET
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
                fut = {ex.submit(worker_compute, j, (n, a, patterns[j], A, R_SET)): j for j in range(len(cols))}
                for ft in as_completed(fut):
                    j, Dr = ft.result()
                    Dr_cols[j] = Dr

            # Build constraint matrix rows for r in CONSTRAINT_RS
            M=[]
            for r in CONSTRAINT_RS:
                row=[Dr_cols[j][r] for j in range(len(cols))]
                den=1
                for x in row:
                    den = (den*x.denominator)//gcd(den, x.denominator)
                M.append([int(x*den) for x in row])

            # Nullspace of constraints
            B = nullspace_basis_q(M)
            if not B:
                print(f"(n={n}, a={a})    no nullspace")
                continue
            if len(B) > MAX_NULLSPACE_DIM:
                B = B[:MAX_NULLSPACE_DIM]

            # scan basis vectors; keep best S-H where constraints vanish and D_target ≠ 0
            best = None
            best_payload = None
            for u in B:
                # exact check: constraints vanish automatically for basis of nullspace of those rows,
                # but numerical guard: verify D_r=0 for r in constraints
                zero_ok = True
                for r in CONSTRAINT_RS:
                    s = Fr(0)
                    for j, cj in enumerate(u):
                        if cj:
                            s += Fr(cj) * Dr_cols[j][r]
                    if s != 0:
                        zero_ok = False
                        break
                if not zero_ok:
                    continue

                # ensure D_target nonzero
                Dt = Fr(0)
                for j, cj in enumerate(u):
                    if cj:
                        Dt += Fr(cj) * Dr_cols[j][TARGET_R]
                if Dt == 0:
                    continue

                S, H, gap, D, L = compute_S_H(u, Dr_cols, Z, TARGET_R)
                if (best is None) or (gap > best):
                    best = gap
                    u_max = max(abs(x) for x in u) if u else 0
                    supp = tuple(i for i,v in enumerate(u) if v)
                    ln_den = float(mp.log(mp.mpf(D.denominator))) if D.denominator != 0 else 0.0
                    ln_L   = float(mp.log(abs(L))) if L != 0 else float("-inf")
                    best_payload = (S, H, gap, ln_den, ln_L, u_max, len(supp), supp)

            if best is not None:
                S,H,gap,ln_den,ln_L,u_max,supp_sz,supp = best_payload
                print(f"(n={n}, a={a})    S-H = {gap:.3f}")
                with open(CSV_PATH, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([TARGET_R, n, a, f"{S:.6f}", f"{H:.6f}", f"{gap:.6f}",
                                f"{ln_den:.6f}", f"{ln_L:.6f}", u_max, supp_sz,
                                ";".join(map(str,supp)), ";".join(map(str,CONSTRAINT_RS))])

    print(f"\nSaved CSV: {CSV_PATH}")

if __name__ == "__main__":
    main()
