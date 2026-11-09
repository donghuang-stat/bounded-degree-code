"""Exact signed injective motif counts for all connected motifs with a given
range of edge numbers.

For a weighted symmetric matrix B with zero diagonal and a motif M on v vertices,

    inj(M, B) = sum over injective maps phi: V(M) -> [n] of prod_{(a,b) in E(M)} B[phi(a), phi(b)].

We use the Mobius inversion over set partitions P of V(M) whose blocks are independent sets,

    inj(M, B) = sum_P mu(P) hom(M / P, B),     mu(P) = prod_blocks (-1)^{|b|-1} (|b|-1)!,

where M / P is a loopless multigraph and an edge of multiplicity k carries the
entrywise power B^{(k)}.  For motifs with at most 9 edges every quotient has
treewidth at most 3.  Quotients of treewidth at most 2 are evaluated by
series-parallel reduction (leaf folding, series reduction, parallel edges);
treewidth-3 quotients are evaluated by conditioning on one vertex, after which the
remainder reduces, using a batched version of the same operations.  All quotients
of all motifs are compiled into one hash-consed DAG of matrix operations, so that
common sub-expressions are evaluated once per input graph.  Costs: O(n^3) per
treewidth-2 node and O(n^4) per batched node, all in BLAS level 3.
"""
import itertools, math, os, pickle, time
import numpy as np
import pynauty

# ----------------------------------------------------------------------------------
# canonical forms
# ----------------------------------------------------------------------------------

def simple_cert(v, edges):
    adj = {i: [] for i in range(v)}
    for a, b in edges:
        adj[a].append(b)
    g = pynauty.Graph(v, directed=False, adjacency_dict=adj)
    return (v, len(edges), pynauty.certificate(g))


def multi_cert(k, med):
    """Canonical key of a loopless multigraph on k vertices; med maps (i, j) with i<j to multiplicity."""
    mults = sorted(set(med.values()))
    N = k + len(med)
    adj = {i: [] for i in range(N)}
    cls = {mu: set() for mu in mults}
    for t, ((i, j), mu) in enumerate(sorted(med.items())):
        x = k + t
        adj[x] = [i, j]
        cls[mu].add(x)
    coloring = [set(range(k))] + [cls[mu] for mu in mults]
    g = pynauty.Graph(N, directed=False, adjacency_dict=adj, vertex_coloring=coloring)
    prof = tuple(sorted(med.values()))
    return (k, prof, pynauty.certificate(g))


# ----------------------------------------------------------------------------------
# enumeration of connected motifs (no isolated vertices) by edge augmentation
# ----------------------------------------------------------------------------------

def connected_motifs_upto(mmax):
    """Return {m: [(v, edges), ...]} for m = 1..mmax, one representative per isomorphism class."""
    out = {1: [(2, [(0, 1)])]}
    level = out[1]
    for m in range(2, mmax + 1):
        seen, new = set(), []
        for v, edges in level:
            es = set(edges)
            cands = [(i, j) for i in range(v) for j in range(i + 1, v) if (i, j) not in es]
            cands += [(i, v) for i in range(v)]
            for e in cands:
                v2 = v + (1 if e[1] == v else 0)
                ed2 = sorted(edges + [e])
                c = simple_cert(v2, ed2)
                if c not in seen:
                    seen.add(c)
                    new.append((v2, ed2))
        out[m] = new
        level = new
    return out


def automorphisms(v, edges):
    adj = {i: [] for i in range(v)}
    for a, b in edges:
        adj[a].append(b)
    g = pynauty.Graph(v, directed=False, adjacency_dict=adj)
    _, grpsize1, grpsize2, _, _ = pynauty.autgrp(g)
    return int(round(grpsize1 * 10 ** grpsize2))


# ----------------------------------------------------------------------------------
# Mobius expansion over independent set partitions
# ----------------------------------------------------------------------------------

def independent_partitions(v, edges):
    nb = [set() for _ in range(v)]
    for a, b in edges:
        nb[a].add(b); nb[b].add(a)
    blk = [-1] * v
    blocks = []

    def rec(x):
        if x == v:
            yield blk
            return
        for bi, B in enumerate(blocks):
            if not (nb[x] & B):
                B.add(x); blk[x] = bi
                yield from rec(x + 1)
                B.discard(x)
        blocks.append({x}); blk[x] = len(blocks) - 1
        yield from rec(x + 1)
        blocks.pop()
    yield from rec(0)


FACT = [math.factorial(i) for i in range(20)]


def mobius_expansion(v, edges, reps):
    """Return {quotient key: coefficient}; reps collects one labeled representative per key."""
    out = {}
    for blk in independent_partitions(v, edges):
        k = max(blk) + 1
        sizes = [0] * k
        for b in blk:
            sizes[b] += 1
        mu = 1
        for s in sizes:
            mu *= (-1) ** (s - 1) * FACT[s - 1]
        med = {}
        for a, b in edges:
            i, j = blk[a], blk[b]
            if i > j:
                i, j = j, i
            med[(i, j)] = med.get((i, j), 0) + 1
        key = multi_cert(k, med)
        if key not in reps:
            reps[key] = (k, dict(med))
        out[key] = out.get(key, 0) + mu
    return {q: c for q, c in out.items() if c != 0}


# ----------------------------------------------------------------------------------
# compilation of homomorphism counts into a shared DAG
# ----------------------------------------------------------------------------------

class Plan:
    """Hash-consed DAG.  Node kinds:
         ('B', mu)                       matrix B^(mu), symmetric
         ('mv', M, t, w)                 vector  x -> sum_y M'[x, y] w[y]   (w may be None)
         ('vp', (w1, w2, ...))           entrywise product of vectors
         ('ser', M1, t1, w, M2, t2)      matrix  M1' diag(w) M2'
         ('had', ((M1, t1), ...))        entrywise product of matrices
         ('rows', M, t)                  batched vector: row i of M' for conditioning value i
         ('sum', w)                      scalar sum_x w[x] (batched if w is)
         ('sp', (s1, s2, ...))           product of scalars (components after conditioning)
         ('dot', s, w)                   scalar sum_i w[i] s[i] over the conditioning vertex
       where M' = M or M^T according to the flag t.  Each node records whether it is batched.
    """

    def __init__(self):
        self.nodes, self.index, self.batched, self.sym = [], {}, [], []

    def _add(self, key, batched, sym=False):
        if key in self.index:
            return self.index[key]
        i = len(self.nodes)
        self.nodes.append(key); self.index[key] = i
        self.batched.append(batched); self.sym.append(sym)
        return i

    def B(self, mu):
        return self._add(('B', mu), False, True)

    def _b(self, *ids):
        return any(self.batched[i] for i in ids if i is not None)

    def norm(self, M, t):
        return (M, 0) if self.sym[M] else (M, t)

    def mv(self, M, t, w):
        M, t = self.norm(M, t)
        return self._add(('mv', M, t, w), self._b(M, w))

    def vp(self, ws):
        ws = [w for w in ws if w is not None]
        if not ws:
            return None
        if len(ws) == 1:
            return ws[0]
        ws = tuple(sorted(ws))
        return self._add(('vp', ws), self._b(*ws))

    def ser(self, M1, t1, w, M2, t2):
        """Return (node, flag) for the oriented matrix M1' diag(w) M2'."""
        a = self.norm(M1, t1); b = self.norm(M2, t2)
        key = ('ser', a[0], a[1], w, b[0], b[1])
        # transpose: M2'^T diag(w) M1'^T
        bt = self.norm(M2, 1 - t2); at = self.norm(M1, 1 - t1)
        keyT = ('ser', bt[0], bt[1], w, at[0], at[1])
        sym = key == keyT
        if keyT < key:
            return self._add(keyT, self._b(M1, M2, w), sym), (0 if sym else 1)
        return self._add(key, self._b(M1, M2, w), sym), 0

    def had(self, terms):
        terms = [self.norm(M, t) for M, t in terms]
        if len(terms) == 1:
            return terms[0]
        key = tuple(sorted(terms))
        keyT = tuple(sorted(self.norm(M, 1 - t) for M, t in terms))
        sym = key == keyT
        Ms = [M for M, _ in terms]
        if keyT < key:
            return self._add(('had', keyT), self._b(*Ms), sym), (0 if sym else 1)
        return self._add(('had', key), self._b(*Ms), sym), 0

    def rows(self, M, t):
        M, t = self.norm(M, t)
        return self._add(('rows', M, t), True)

    def ssum(self, w):
        return self._add(('sum', w), self._b(w))

    def sprod(self, ss):
        if len(ss) == 1:
            return ss[0]
        ss = tuple(sorted(ss))
        return self._add(('sp', ss), self._b(*ss))

    def dot(self, s, w):
        return self._add(('dot', s, w), False)

    # ---------------------------------------------------------------------------
    def _reduce(self, verts, wt, E):
        """Series-parallel reduction.  verts: set; wt: {v: vec or None};
        E: {(u, x): (node, flag)} oriented as [u, x], stored for both orientations.
        Returns the final scalar node or None if the graph is not reducible."""
        def nbrs(x):
            return [y for y in verts if (x, y) in E]
        factors = []
        while len(verts) > 1:
            cand = None
            for x in sorted(verts):
                d = len(nbrs(x))
                if d <= 2 and (cand is None or d < cand[0]):
                    cand = (d, x)
                    if d <= 1:
                        break
            if cand is None:
                return None
            d, x = cand
            nx_ = nbrs(x)
            if d == 0:
                factors.append(self.ssum(wt[x]))
            elif d == 1:
                u = nx_[0]
                M, t = E[(u, x)]
                wt[u] = self.vp([wt[u], self.mv(M, t, wt[x])])
                del E[(u, x)], E[(x, u)]
            else:
                u, y = nx_
                M1, t1 = E[(u, x)]
                M2, t2 = E[(x, y)]
                S = self.ser(M1, t1, wt[x], M2, t2)
                if (u, y) in E:
                    S = self.had([E[(u, y)], S])
                E[(u, y)] = S
                E[(y, u)] = (S[0], 1 - S[1]) if not self.sym[S[0]] else (S[0], 0)
                del E[(u, x)], E[(x, u)], E[(x, y)], E[(y, x)]
            verts.discard(x); del wt[x]
        (x,) = verts
        factors.append(self.ssum(wt[x]))
        return self.sprod(factors)

    def compile_quotient(self, k, med):
        def fresh():
            E = {}
            for (i, j), mu in med.items():
                b = self.B(mu)
                E[(i, j)] = (b, 0); E[(j, i)] = (b, 0)
            return set(range(k)), {x: None for x in range(k)}, E
        # quick structural test: is the underlying simple graph reducible?
        if self._reducible(k, med, None):
            verts, wt, E = fresh()
            return self._reduce(verts, wt, E), 3
        for a in range(k):
            if self._reducible(k, med, a):
                verts, wt, E = fresh()
                for x in range(k):
                    if x != a and (a, x) in E:
                        M, t = E[(a, x)]
                        wt[x] = self.vp([wt[x], self.rows(M, t)])
                        del E[(a, x)], E[(x, a)]
                verts.discard(a); del wt[a]
                s = self._reduce(verts, wt, E)
                return self.dot(s, None), 4
        raise RuntimeError("quotient of treewidth > 3 after one conditioning")

    @staticmethod
    def _reducible(k, med, drop):
        adj = {x: set() for x in range(k) if x != drop}
        for (i, j) in med:
            if drop in (i, j):
                continue
            adj[i].add(j); adj[j].add(i)
        while len(adj) > 1:
            x = next((x for x in adj if len(adj[x]) <= 2), None)
            if x is None:
                return False
            nb = list(adj[x])
            for y in nb:
                adj[y].discard(x)
            if len(nb) == 2:
                adj[nb[0]].add(nb[1]); adj[nb[1]].add(nb[0])
            del adj[x]
        return True


# ----------------------------------------------------------------------------------
# library
# ----------------------------------------------------------------------------------

class MotifLibrary:
    def __init__(self, mmin=3, mmax=8, cache_dir=None, verbose=False):
        cache = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache = os.path.join(cache_dir, f"motiflib_{mmin}_{mmax}.pkl")
        if cache and os.path.exists(cache):
            with open(cache, "rb") as f:
                self.__dict__.update(pickle.load(f))
            return
        t0 = time.time()
        allm = connected_motifs_upto(mmax)
        self.motifs = []
        for m in range(mmin, mmax + 1):
            for idx, (v, edges) in enumerate(allm[m]):
                deg = [0] * v
                for a, b in edges:
                    deg[a] += 1; deg[b] += 1
                self.motifs.append(dict(name=f"E{m}_{idx}", m=m, v=v, edges=edges,
                                        aut=automorphisms(v, edges), maxdeg=max(deg),
                                        tree=(v == m + 1), cycle=(v == m and max(deg) == 2)))
        if verbose:
            print(f"[lib] {len(self.motifs)} motifs enumerated ({time.time()-t0:.1f}s)", flush=True)
        reps, exps = {}, []
        for M in self.motifs:
            exps.append(mobius_expansion(M['v'], M['edges'], reps))
        keys = sorted({q for e in exps for q in e}, key=lambda q: (sum(q[1]), q[0], q[1], q[2]))
        kidx = {q: i for i, q in enumerate(keys)}
        rows, cols, vals = [], [], []
        for a, e in enumerate(exps):
            for q, c in e.items():
                rows.append(a); cols.append(kidx[q]); vals.append(c)
        self.coef = (np.array(rows), np.array(cols), np.array(vals, dtype=float))
        self.nq = len(keys)
        if verbose:
            print(f"[lib] {self.nq} distinct quotients ({time.time()-t0:.1f}s)", flush=True)
        plan = Plan()
        self.qnode, self.qtw = [], []
        for q in keys:
            k, med = reps[q]
            node, tw = plan.compile_quotient(k, med)
            self.qnode.append(node); self.qtw.append(tw)
        self.nodes, self.batched = plan.nodes, plan.batched
        self._schedule()
        if verbose:
            nb = sum(self.batched)
            print(f"[lib] DAG: {len(self.nodes)} nodes ({nb} batched); "
                  f"treewidth<=2 quotients {self.qtw.count(3)}, treewidth-3 quotients {self.qtw.count(4)} "
                  f"({time.time()-t0:.1f}s)", flush=True)
        if cache:
            # atomic write: concurrent readers (e.g. many array tasks starting at once)
            # see either no cache file or a complete one, never a partial pickle
            tmp = f"{cache}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
            try:
                with open(tmp, "xb") as f:
                    pickle.dump({k: v for k, v in self.__dict__.items()}, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, cache)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise

    # children of a node
    @staticmethod
    def _children(key):
        kind = key[0]
        if kind == 'B':
            return []
        if kind == 'mv':
            return [key[1]] + ([key[3]] if key[3] is not None else [])
        if kind == 'vp':
            return list(key[1])
        if kind == 'ser':
            return [key[1], key[4]] + ([key[3]] if key[3] is not None else [])
        if kind == 'had':
            return [M for M, _ in key[1]]
        if kind == 'rows':
            return [key[1]]
        if kind == 'sum':
            return [key[1]] if key[1] is not None else []
        if kind == 'dot':
            return [key[1]] + ([key[2]] if key[2] is not None else [])
        if kind == 'sp':
            return list(key[1])
        raise ValueError(kind)

    def _schedule(self):
        N = len(self.nodes)
        # phase 1: unbatched nodes; phase 2: batched nodes and 'dot' nodes
        self.phase2 = [i for i in range(N) if self.batched[i] or self.nodes[i][0] == 'dot']
        p2 = set(self.phase2)
        self.phase1 = [i for i in range(N) if i not in p2]
        keep = set()
        for i in self.phase2:
            for c in self._children(self.nodes[i]):
                if not self.batched[c]:
                    keep.add(c)
        last = {}
        for pos, i in enumerate(self.phase1):
            for c in self._children(self.nodes[i]):
                last[c] = pos
        self.free_after = {}
        for c, pos in last.items():
            if c not in keep:
                self.free_after.setdefault(pos, []).append(c)
        last2 = {}
        for pos, i in enumerate(self.phase2):
            for c in self._children(self.nodes[i]):
                last2[c] = pos
        self.free_after2 = {}
        for c, pos in last2.items():
            if self.batched[c]:
                self.free_after2.setdefault(pos, []).append(c)
        self.keep = keep

    # ---------------------------------------------------------------------------
    def _eval(self, key, val, n, chunk):
        kind = key[0]
        if kind == 'mv':
            M = val[key[1]]; M = np.swapaxes(M, -1, -2) if key[2] else M
            w = val[key[3]] if key[3] is not None else None
            if w is None:
                return M.sum(-1)
            if M.ndim == 2:
                return M @ w if w.ndim == 1 else w @ M.T
            return np.einsum('bij,bj->bi', M, w) if w.ndim == 2 else M @ w
        if kind == 'vp':
            out = val[key[1][0]]
            for j in key[1][1:]:
                out = out * val[j]
            return out
        if kind == 'ser':
            L = val[key[1]]; L = np.swapaxes(L, -1, -2) if key[2] else L
            if key[3] is not None:
                L = L * val[key[3]][..., None, :]
            R = val[key[4]]; R = np.swapaxes(R, -1, -2) if key[5] else R
            if R.ndim == 2:
                if L.ndim == 2:
                    return L @ R
                b = L.shape[0]
                return (np.ascontiguousarray(L).reshape(b * n, n) @ R).reshape(b, n, n)
            return np.matmul(L, R)
        if kind == 'had':
            out = None
            for M, t in key[1]:
                X = val[M]; X = np.swapaxes(X, -1, -2) if t else X
                out = X if out is None else out * X
            return out
        if kind == 'rows':
            X = val[key[1]]; X = X.T if key[2] else X
            return X[chunk[0]:chunk[1]]
        if kind == 'sum':
            return val[key[1]].sum(-1) if key[1] is not None else float(n)
        if kind == 'sp':
            out = val[key[1][0]]
            for j in key[1][1:]:
                out = out * val[j]
            return out
        raise ValueError(kind)

    def homs(self, B, chunk_size=None):
        n = B.shape[0]
        if chunk_size is None:
            chunk_size = max(1, min(n, int(2.0e7 // (n * n))))   # ~160 MB per batched matrix
        val = {}
        pw = {1: B}
        nodes = self.nodes
        for pos, i in enumerate(self.phase1):
            key = nodes[i]
            if key[0] == 'B':
                mu = key[1]
                if mu not in pw:
                    pw[mu] = B ** mu
                val[i] = pw[mu]
            else:
                val[i] = self._eval(key, val, n, None)
            for c in self.free_after.get(pos, ()):
                val.pop(c, None)
        if self.phase2:
            acc = {}
            for c0 in range(0, n, chunk_size):
                ch = (c0, min(n, c0 + chunk_size))
                for pos, i in enumerate(self.phase2):
                    key = nodes[i]
                    if key[0] == 'dot':
                        s = val[key[1]]
                        w = val[key[2]][ch[0]:ch[1]] if key[2] is not None else None
                        acc[i] = acc.get(i, 0.0) + float(s.sum() if w is None else s @ w)
                    else:
                        val[i] = self._eval(key, val, n, ch)
                    for c in self.free_after2.get(pos, ()):
                        val.pop(c, None)
            val.update(acc)
        return np.array([float(val[i]) for i in self.qnode])

    def inj(self, B, **kw):
        h = self.homs(B, **kw)
        r, c, v = self.coef
        out = np.zeros(len(self.motifs))
        np.add.at(out, r, v * h[c])
        return out


def inj_bruteforce(v, edges, B):
    n = B.shape[0]
    tot = 0.0
    for phi in itertools.permutations(range(n), v):
        pr = 1.0
        for a, b in edges:
            pr *= B[phi[a], phi[b]]
            if pr == 0.0:
                break
        tot += pr
    return tot


def sample_pair(rng, n, p, rho):
    """Correlated Erdos-Renyi pair with identity latent permutation (the statistic is
    invariant under relabeling each graph separately)."""
    m = n * (n - 1) // 2
    q = p * (1 - p); p11 = p * p + rho * q; p10 = q * (1 - rho)
    u = rng.random(m)
    x = u < p11 + p10
    y = (u < p11) | ((u >= p11 + p10) & (u < p11 + 2 * p10))
    iu = np.triu_indices(n, 1)
    A1 = np.zeros((n, n)); A2 = np.zeros((n, n))
    A1[iu] = x; A2[iu] = y
    return A1 + A1.T, A2 + A2.T


def centered(A, p):
    B = A - p
    np.fill_diagonal(B, 0.0)
    return B


def log_norm_weight(lib, n, p):
    """log of (n-v)!/(n! (p(1-p))^e aut(M)) so that weight * inj1 * inj2 has null variance one."""
    q = p * (1 - p)
    return np.array([math.lgamma(n - M['v'] + 1) - math.lgamma(n + 1) - math.log(M['aut']) - M['m'] * math.log(q)
                     for M in lib.motifs])
