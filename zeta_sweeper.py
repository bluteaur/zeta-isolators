#!/usr/bin/env python3
# HIGH-N SWEEP for promising zetas (focused on ζ(15))

import sys
sys.set_int_max_str_digits(0)   # 0 = unlimited
from fractions import Fraction as Fr
from math import gcd
import itertools, os, csv, random, time
import mpmath as mp
import numpy as np

# ================== CONFIG ==================
mp.mp.dps   = 350
CSV_PATH    = "high_n_results2.csv"

# FOCUSED TARGETS - just the most promising
TARGETS     = [9, 11, 13, 15, 17, 19, 21]  # Starting with ζ(15) - best gap so far

# HIGH-N sweep - push to n=60

#Testing n=a*2.7 and n=a*2.71 as integers (faster than fractional)
#BASE_PAIRS = [(27, 10)]
BASE_PAIRS = [(271, 100)]

#testing fraction n = a/e (slower)
#BASE_PAIRS  = [(x, x/np.e) for x in range(5, 10)]

#Using ~3 or pi, more consistent gap, int is faster
#BASE_PAIRS  = [(int(np.pi*x), x) for x in range(20, 29)]
#BASE_PAIRS  = [(x*3, x) for x in range(20, 29)]

#Random values
#BASE_PAIRS = [
#    (38, 13), (40, 13), (42, 14), (44, 15),
#    (46, 15), (48, 16), (50, 17), (52, 17),
#    (54, 18), (56, 19), (58, 19), (60, 20)
#]

A_FROM_R = lambda r: max(int(1.3 * r), 12)
ALT_A_OFFS  = [0, +2]          # Reduced: just A and A+2
ALT_A_FILL  = [-4, -6]         # Reduced: just top 2 performers

MAX_FAMS        = 8            # Reduced for speed
RANDOM_FAM_TRY  = 24           # Reduced for speed
COMB_RANGE      = list(range(-6,-0)) + list(range(1,7))
MAX_RANDOM_COMB = 5000        # Reduced for speed
MAX_BASIS_ENUM  = 800
NULL_REPICKS    = 3
ALPHA_JITTER    = [-1,0,1,2]

UNWANTED_LADDER = [
    (3,5,7,9),
    (3,5,7),
    (3,5),
    (3,),
    tuple(),
]
PENALTY_WEIGHTS = {3:0.02, 5:0.02, 7:0.02, 9:0.02}

# ================== POLY / SERIES CORE ==================
def poly_mul(p,q):
    r=[Fr(0)]*(len(p)+len(q)-1)
    for i,a in enumerate(p):
        if a:
            for j,b in enumerate(q):
                if b:
                    r[i+j]+=a*b
    return r

def rising(b,n):
    r=[Fr(1)]
    for j in range(n):
        r=poly_mul(r,[Fr(b+j),Fr(1)])
    return r

def invert_series(p,deg):
    inv0=Fr(1)/p[0]
    out=[inv0]
    for k in range(1,deg+1):
        s=Fr(0)
        for i in range(1,k+1):
            if i<len(p) and k-i<len(out):
                s+=p[i]*out[k-i]
        out.append(-inv0*s)
    return out

def mul_series(p,q,deg):
    r=[Fr(0)]*(deg+1)
    for i,a in enumerate(p):
        if a:
            for j,b in enumerate(q):
                if b and i+j<=deg:
                    r[i+j]+=a*b
    return r

def val_series(p):
    for i,c in enumerate(p):
        if c!=0: return i
    return len(p)

def Dr_exact(n,a,alphas,A):
    P = rising(0,n)
    P5 = P
    for _ in range(4): P5 = poly_mul(P5, P)

    Num = [Fr(1)]
    for α in alphas:
        Num = poly_mul(Num, rising(α - a, n))
    Num = [Fr(0)] + Num

    deg = 2*A - 1
    v = val_series(P5)
    Den = P5[v:]
    Num2 = Num[v:]
    H = mul_series(Num2, invert_series(Den, deg), deg)

    out = {}
    for r in range(3, 2*A+2, 2):
        idx = (2*A+2) - r
        out[r] = H[idx] if 0 <= idx <= deg else Fr(0)
    return out

# ================== RATIONAL RREF / NULLSPACE ==================
def rref_q(M):
    R=[[Fr(x) for x in row] for row in M]
    rows=len(R); cols=len(R[0])
    piv=[]; r=0
    for c in range(cols):
        pr=None
        for i in range(r,rows):
            if R[i][c]!=0: pr=i; break
        if pr is None: continue
        R[r],R[pr]=R[pr],R[r]
        inv=Fr(1)/R[r][c]
        R[r]=[inv*x for x in R[r]]
        for i in range(rows):
            if i!=r and R[i][c]!=0:
                fac=R[i][c]
                R[i]=[R[i][j]-fac*R[r][j] for j in range(cols)]
        piv.append((r,c)); r+=1
        if r==rows: break
    return R,piv

def nullspace_basis(M):
    if not M:
        return [[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]
    R,piv=rref_q(M)
    rows=len(R); cols=len(R[0])
    pivc={c for _,c in piv}
    free=[j for j in range(cols) if j not in pivc]
    B=[]
    for f in free:
        v=[Fr(0)]*cols; v[f]=Fr(1)
        for r,c in reversed(piv):
            s=Fr(0)
            for j in range(c+1,cols):
                s+=R[r][j]*v[j]
            v[c] = -s
        den=1
        for x in v:
            den=(den*x.denominator)//gcd(den,x.denominator)
        ints=[int(x*den) for x in v]
        g=0
        for t in ints: g=gcd(g,abs(t))
        if g>1: ints=[t//g for t in ints]
        B.append(ints)
    return B

# ================== UTIL ==================
def mpf(fr: Fr):
    return mp.mpf(fr.numerator)/mp.mpf(fr.denominator) if fr!=0 else mp.mpf(0)

def lnabs(x): return float(mp.log(abs(x))) if x!=0 else float("-inf")

def gcd_reduce_vec(v):
    g=0
    for t in v: g=gcd(g,abs(t))
    if g>1 and g!=0: v=[t//g for t in v]
    return v

# ================== ALPHA FAMILIES ==================
def alpha_family_structured(A, fill, fam_id):
    base=[fill]*A
    def t(x): return tuple(base[:-1]+[x])
    fams = [
        (tuple(base), t(fill-1), t(fill-2), t(fill-3), tuple(base[:-2]+[fill-1,fill-1])),
        (tuple(base), t(fill-2), t(fill-4), t(fill-5), t(fill-1)),
        (tuple(base), tuple(base[:-2]+[fill-3,fill-2]), tuple(base[:-2]+[fill-2,fill-2]), tuple(base[:-2]+[fill-4,fill-3]), t(fill-5)),
        (tuple(base), tuple(base[:-2]+[fill-5]), tuple(base[:-2]+[fill-3]), tuple(base[:-2]+[fill-2]), t(fill-1)),
        (t(fill-6), t(fill-7), tuple(base[:-2]+[fill-6,fill-5]), tuple(base[:-2]+[fill-7]), tuple(base[:-2]+[fill-8])),
        (tuple(base[:-3]+[fill-5,fill-4,fill-3]), tuple(base[:-3]+[fill-4,fill-5,fill-3]),
         tuple(base[:-3]+[fill-4,fill-4,fill-5]), tuple(base[:-2]+[fill-6]), tuple(base[:-2]+[fill-4,fill-4])),
        (tuple(base[:-1]+[fill-6]), tuple(base[:-1]+[fill-7]), t(fill-8), tuple(base[:-2]+[fill-7]), tuple(base[:-2]+[fill-9])),
        (tuple(base[:-1]+[fill-10]), tuple(base[:-1]+[fill-11]), tuple(base[:-2]+[fill-10,fill-9]),
         tuple(base[:-2]+[fill-9,fill-7]), tuple(base[:-2]+[fill-12,fill-8])),
    ]
    fam = fams[fam_id % len(fams)]
    return [tuple(col) for col in fam]

def alpha_family_random(A, fill, rng):
    base=[fill]*A
    cols=[]
    for _ in range(5):
        col=list(base)
        for j in range(max(1, A//3), A):
            col[j] += rng.choice(ALPHA_JITTER) - rng.randint(0,2)
        col[-1] += rng.randint(-6, -1)
        cols.append(tuple(col))
    return cols

# ================== BUILD / EVAL ==================
def build_constraint_matrix(Dr_cols, unwanted):
    M=[]
    for rr in unwanted:
        row=[Dr_cols[j].get(rr,Fr(0)) for j in range(5)]
        if all(v==0 for v in row):
            continue
        den=1
        for x in row:
            den=(den*x.denominator)//gcd(den,x.denominator)
        M.append([int(x*den) for x in row])
    return M

def eval_u_and_score(u, Dr_cols, target_r, penalty_rs):
    D={}
    maxR = max(max(d.keys()) for d in Dr_cols)
    for r in range(3, maxR+1, 2):
        s=Fr(0)
        for j,w in enumerate(u):
            s += Fr(w)*Dr_cols[j].get(r,Fr(0))
        D[r]=s

    Dr = D.get(target_r, Fr(0))
    if Dr == 0:
        return None

    L = mp.mpf(0)
    for r,fr in D.items():
        if fr!=0:
            L += mpf(fr)*mp.zeta(r)

    lnD = lnabs(mpf(Dr))
    lnL = lnabs(L)
    gap = lnD - lnL

    penalty = 0.0
    for rr in penalty_rs:
        fr = D.get(rr, Fr(0))
        if fr != 0:
            penalty -= PENALTY_WEIGHTS.get(rr, 0.0) * lnabs(mpf(fr))
    score = gap + penalty

    return {
        "lnD": lnD, "lnL": lnL, "gap": gap, "score": score,
        "supp": sum(1 for w in u if w), "maxu": max(abs(w) for w in u),
        "u": tuple(u), "D": D
    }

def try_basis_combinations(basis, Dr_cols, target_r, penalty_rs, rng):
    best=None
    enum = itertools.product(COMB_RANGE, repeat=len(basis))
    for k, coeffs in enumerate(enum):
        if k > MAX_BASIS_ENUM: break
        if all(c==0 for c in coeffs): continue
        u=[0]*5
        for c,b in zip(coeffs,basis):
            if c:
                u=[ui + c*bi for ui,bi in zip(u,b)]
        u=gcd_reduce_vec(u)
        if all(t==0 for t in u): continue
        res = eval_u_and_score(u, Dr_cols, target_r, penalty_rs)
        if res is None: continue
        if (best is None) or (res["score"] > best["score"]):
            best=res
    for _ in range(MAX_RANDOM_COMB):
        u=[0]*5
        for b in basis:
            c = rng.choice(COMB_RANGE + [0,0,0])
            if c:
                u=[ui + c*bi for ui,bi in zip(u,b)]
        u=gcd_reduce_vec(u)
        if all(t==0 for t in u): continue
        res = eval_u_and_score(u, Dr_cols, target_r, penalty_rs)
        if res is None: continue
        if (best is None) or (res["score"] > best["score"]):
            best=res
    return best

def run_one(n,a,target_r,rng):
    for Aoff in ALT_A_OFFS:
        A0 = A_FROM_R(target_r) + Aoff
        if A0 < 6: continue
        for fill in ALT_A_FILL:
            for fam in range(MAX_FAMS):
                alphas_cols = alpha_family_structured(A0, fill, fam)
                Dr_cols = [Dr_exact(n,a,alph,A0) for alph in alphas_cols]

                for unwanted in UNWANTED_LADDER:
                    M = build_constraint_matrix(Dr_cols, unwanted)
                    basis = nullspace_basis(M)
                    best = try_basis_combinations(
                        basis, Dr_cols, target_r,
                        penalty_rs=set((3,5,7,9)) - set(unwanted), rng=rng
                    )
                    if best is not None:
                        return best, dict(n=n,a=a,A=A0,fam=f"str:{fam}",unwanted=tuple(unwanted),fill=fill)

            for _ in range(RANDOM_FAM_TRY):
                alphas_cols = alpha_family_random(A0, fill, rng)
                Dr_cols = [Dr_exact(n,a,alph,A0) for alph in alphas_cols]
                for unwanted in UNWANTED_LADDER:
                    M = build_constraint_matrix(Dr_cols, unwanted)
                    basis = nullspace_basis(M)
                    best = try_basis_combinations(
                        basis, Dr_cols, target_r,
                        penalty_rs=set((3,5,7,9)) - set(unwanted), rng=rng
                    )
                    if best is not None:
                        return best, dict(n=n,a=a,A=A0,fam="rnd",unwanted=tuple(unwanted),fill=fill)

    raise RuntimeError("no nonzero D_target found")

# ================== CSV ==================
def ensure_csv(path=CSV_PATH):
    exists = os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if not exists:
        w.writerow([
            "target","n","a","A","family","constraints","fill",
            "lnD","lnL","gap","score","supp","maxu"
        ])
    return f,w

# ================== MAIN ==================
def main():
    rng = random.Random(1337)
    f,w = ensure_csv(CSV_PATH)
    t0=time.time()

    for r in TARGETS:
        print(f"\n{'='*70}")
        print(f"HIGH-N SWEEP FOR ζ({r})")
        print(f"{'='*70}\n")
        
        for (n,a) in BASE_PAIRS:
            t_start = time.time()
            try:
                best, meta = run_one(n,a,r,rng)
                elapsed = time.time() - t_start
                print(f"(n={meta['n']:2d}, a={meta['a']:2f})   "
                      f"gap={best['gap']:7.3f}   "
                      f"fam={meta['fam']:6s}   "
                      f"fill={meta['fill']:3d}   "
                      f"[{elapsed:5.1f}s]")
                w.writerow([
                    r, meta["n"], meta["a"], meta["A"], meta["fam"], " ".join(map(str,meta["unwanted"])),
                    meta["fill"], f"{best['lnD']:.6f}", f"{best['lnL']:.6f}", f"{best['gap']:.6f}", f"{best['score']:.6f}",
                    best["supp"], best["maxu"]
                ])
                f.flush()
            except Exception as e:
                print(f"(n={n:2d}, a={a:2f})   ERROR: {e}")
                
    print(f"\n{'='*70}")
    print(f"COMPLETED in {time.time()-t0:.1f}s")
    print(f"Results saved to: {CSV_PATH}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
