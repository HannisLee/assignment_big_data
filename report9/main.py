"""
L9 Assignment — Fastmap: Achieving O(n m^2) via Lazy (On-Demand) Deflation

The naive Fastmap algorithm rewrites the *entire* n x n distance matrix every
iteration (the eager `Deflate` step), giving O(n^2 m) total time. This script
shows that the deflation can be done *lazily*: the residual distance is computed
on demand from the original distance matrix d0 and the coordinates already
assigned. Since Fastmap only ever queries O(n) distances per iteration, the
redundant O(n^2) work disappears and the claimed O(n m^2) is recovered.

We implement both versions, verify they produce identical embeddings, and
generate the six report figures.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
import time

# ── Palette (consistent with earlier reports) ────────────────────────────────
C_BLUE   = "#3498db"
C_ORANGE = "#e67e22"
C_GREEN  = "#1e8449"
C_RED    = "#c0392b"
C_PURPLE = "#8e44ad"
C_GREY   = "#7f8c8d"
C_DARK   = "#2c3e50"
BLOB_COLORS = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE]


def banner(title):
    print("=" * 74)
    print(title)
    print("=" * 74)


# ════════════════════════════════════════════════════════════════════════════
#  Fastmap — NAIVE   O(n^2 m) :  eager, full-matrix deflation
# ════════════════════════════════════════════════════════════════════════════
def _fastmap_naive_core(D02, m, want_snaps=False):
    """Core loop on already-squared distances D02. Eager full-matrix deflation.

    The O(n^2) `Deflate` rewrite of the *whole* matrix, every dimension, is the
    work we isolate here — it is exactly the O(n^2 m) cost the assignment asks
    us to remove. (Squaring the input is excluded: it is O(n^2) for *every*
    algorithm and is not what the question is about.)
    """
    D2 = D02.copy()                               # working squared-distance matrix
    n = D02.shape[0]
    X = np.zeros((n, m))
    snaps = [np.sqrt(np.maximum(D2, 0.0)).copy()] if want_snaps else None
    for i in range(m):
        A = int(np.argmax(D2[0, :]))              # farthest from object 0
        B = int(np.argmax(D2[A, :]))              # farthest from A  -> pivot pair
        dAB2 = D2[A, B]
        dAB = np.sqrt(dAB2)
        if dAB < 1e-12:
            break
        X[:, i] = (D2[A, :] + dAB2 - D2[B, :]) / (2.0 * dAB)   # project onto A-B
        # ── EAGER deflation: O(n^2), touches EVERY pair ──
        diff = X[:, i][:, None] - X[:, i][None, :]
        D2 = D2 - diff ** 2
        D2 = np.maximum(D2, 0.0)
        if want_snaps:
            snaps.append(np.sqrt(D2).copy())
    return X, snaps


def fastmap_naive(D0, m, want_snaps=False):
    """Original Fastmap: O(n^2 m). Accepts raw distances, squares internally."""
    return _fastmap_naive_core((D0.astype(float)) ** 2, m, want_snaps)


# ════════════════════════════════════════════════════════════════════════════
#  Fastmap — EFFICIENT   O(n m^2) :  lazy, on-demand deflation
# ════════════════════════════════════════════════════════════════════════════
def _residual_row(D02, X, i, p):
    """Return d_{i-1}(p, c)^2 for every c, derived on demand.  Cost O(n * i).

    The original squared distances are kept read-only; we only subtract the
    squared projection differences of the dims already assigned (1 .. i-1).
    """
    res = D02[p, :].copy()
    for j in range(i):
        res -= (X[p, j] - X[:, j]) ** 2
    return np.maximum(res, 0.0)


def _fastmap_efficient_core(D02, m):
    """Core loop on already-squared distances D02. No Deflate pass.

    Residual distances are derived from d0 + X on demand, so only O(n) distance
    rows are ever touched per iteration — the O(n m^2) work we want to measure.
    """
    n = D02.shape[0]
    X = np.zeros((n, m))
    for i in range(m):
        row0 = _residual_row(D02, X, i, 0)
        A = int(np.argmax(row0))                   # farthest from object 0
        rowA = _residual_row(D02, X, i, A)
        B = int(np.argmax(rowA))                   # farthest from A  -> pivot pair
        dAB2 = rowA[B]
        dAB = np.sqrt(dAB2)
        if dAB < 1e-12:
            break
        rowB = _residual_row(D02, X, i, B)
        X[:, i] = (rowA + dAB2 - rowB) / (2.0 * dAB)
        # ── NO deflation pass: the matrix is never rewritten ──
    return X


def fastmap_efficient(D0, m):
    """Efficient Fastmap: O(n m^2). Accepts raw distances, squares internally."""
    return _fastmap_efficient_core((D0.astype(float)) ** 2, m)


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════
def make_blobs(n_per=20, centers=3, seed=7):
    rng = np.random.default_rng(seed)
    pts, lbl = [], []
    loci = np.array([[0, 0], [6, 1], [3, 6]][:centers], dtype=float)
    for k in range(centers):
        pts.append(rng.normal(loc=loci[k], scale=0.8, size=(n_per, 2)))
        lbl += [k] * n_per
    return np.vstack(pts), np.array(lbl)


def pairwise_dist(P):
    """Euclidean distance matrix, computed memory-efficiently via the binomial."""
    G = (P ** 2).sum(1)
    D2 = G[:, None] + G[None, :] - 2.0 * (P @ P.T)
    return np.sqrt(np.maximum(D2, 0.0))


def procrustes_to(true_pts, embed):
    """Rigidly align `embed` onto `true_pts` (centre + optimal rotation/reflection)."""
    t = true_pts - true_pts.mean(0)
    e = embed - embed.mean(0)
    U, _, Vt = np.linalg.svd(t.T @ e)
    R = U @ Vt
    if np.linalg.det(R) < 0:                       # allow reflection
        Vt[-1] *= -1
        R = U @ Vt
    return e @ R.T


# ════════════════════════════════════════════════════════════════════════════
#  Figure 1 — the geometry of deflation
# ════════════════════════════════════════════════════════════════════════════
def fig_deflation_geometry():
    fig, ax = plt.subplots(figsize=(10, 5.8))
    xA, xB = -4.0, 4.0
    ax.plot([xA, xB], [0, 0], color=C_DARK, lw=3.4, zorder=2)            # pivot axis
    ax.plot([xA, xB], [0, 0], "o", color=C_DARK, ms=10, zorder=4)        # pivots A, B

    C = np.array([-1.7, 2.2]);  D = np.array([2.5, 1.0])                 # two objects
    Q = np.array([D[0], C[1]])                                           # right-angle corner

    ax.add_patch(Polygon([C, Q, D], closed=True, facecolor=C_BLUE,
                         alpha=0.12, zorder=1))                          # right triangle
    for P in (C, D):                                                     # projections
        ax.plot([P[0], P[0]], [0, P[1]], ls="--", color=C_GREY, lw=1.4, zorder=2)
        ax.plot(P[0], 0, "o", color=C_DARK, ms=6, zorder=3)
    ax.plot([C[0], D[0]], [C[1], D[1]], color=C_RED, lw=2.8, zorder=3)   # hypotenuse d(C,D)
    ax.plot([C[0], Q[0]], [C[1], Q[1]], color=C_GREEN, lw=2.4, zorder=3) # horizontal leg
    ax.plot([Q[0], D[0]], [Q[1], D[1]], color=C_ORANGE, lw=2.4, zorder=3)# vertical leg
    ax.add_patch(Rectangle((Q[0] - 0.17, Q[1] - 0.17), 0.34, 0.34, fill=False,
                           ec=C_DARK, lw=1.5, zorder=3))                 # right-angle mark

    ax.text(C[0] - 0.12, C[1] + 0.22, "C", fontsize=14, fontweight="bold",
            color=C_DARK, ha="right")
    ax.text(D[0] + 0.12, D[1] + 0.22, "D", fontsize=14, fontweight="bold", color=C_DARK)
    ax.text(xA, -0.55, "A", fontsize=14, fontweight="bold", color=C_DARK, ha="center")
    ax.text(xB, -0.55, "B", fontsize=14, fontweight="bold", color=C_DARK, ha="center")
    ax.text(0, -0.95, "pivot axis  (dimension i)", fontsize=11.5, ha="center",
            style="italic", color=C_DARK)
    ax.text((C[0] + Q[0]) / 2, C[1] + 0.26, r"$x_C - x_D$", fontsize=12.5,
            color=C_GREEN, fontweight="bold", ha="center")
    ax.text(Q[0] + 0.20, (Q[1] + D[1]) / 2, "residual", fontsize=12.5,
            color=C_ORANGE, fontweight="bold", va="center", rotation=90)
    ax.text((C[0] + D[0]) / 2 + 0.05, (C[1] + D[1]) / 2 + 0.20, r"$d(C,D)$",
            fontsize=13.5, color=C_RED, fontweight="bold")

    ax.text(0.0, 3.55,
            r"$d(C,D)^{2} \;=\; (x_C - x_D)^{2} \;+\; \mathrm{residual}^{2}$"
            "\n"
            r"$\Rightarrow\;\; d_{i}(C,D)^{2} \;=\; d_{0}(C,D)^{2}"
            r" \;-\; \sum_{j=1}^{i}(x_{C,j}-x_{D,j})^{2}$",
            fontsize=12.5, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.55", fc="#fdf6e3", ec=C_DARK, lw=1.3))
    ax.set_xlim(-4.6, 4.6); ax.set_ylim(-1.4, 4.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Why deflation works — peeling off one pivot dimension at a time",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("deflation_geometry.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved deflation_geometry.png")


# ════════════════════════════════════════════════════════════════════════════
#  Figure 2 — Fastmap recovers structure from distances alone
# ════════════════════════════════════════════════════════════════════════════
def fig_embedding(true_pts, labels, X_eff):
    aligned = procrustes_to(true_pts, X_eff)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, P, ttl in [(axes[0], true_pts, "True coordinates (ground truth)"),
                       (axes[1], aligned, "Fastmap 2-D embedding (from distances only)")]:
        for k in np.unique(labels):
            mm = labels == k
            ax.scatter(P[mm, 0], P[mm, 1], s=58, color=BLOB_COLORS[k],
                       edgecolor="white", linewidth=1.1, label=f"blob {k + 1}", zorder=3)
        ax.set_title(ttl, fontsize=12.5, fontweight="bold")
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
        ax.legend(fontsize=10); ax.grid(alpha=0.25); ax.set_aspect("equal")
    fig.suptitle("Fastmap recovers the 3-blob structure from the pairwise distance matrix alone",
                 fontsize=13.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig("fastmap_embedding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved fastmap_embedding.png")


# ════════════════════════════════════════════════════════════════════════════
#  Figure 3 — naive and efficient produce identical coordinates
# ════════════════════════════════════════════════════════════════════════════
def fig_parity(Xn, Xe):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7))
    for ax, k, name in [(axes[0], 0, "dim 1"), (axes[1], 1, "dim 2")]:
        lo = min(Xn[:, k].min(), Xe[:, k].min())
        hi = max(Xn[:, k].max(), Xe[:, k].max())
        pad = 0.08 * (hi - lo + 1e-9)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], ls="--", color=C_GREY, lw=1.6, label="y = x")
        ax.scatter(Xn[:, k], Xe[:, k], s=48, color=C_BLUE,
                   edgecolor="white", linewidth=1.0, zorder=3)
        ax.set_xlabel(f"naive coordinate  ({name})", fontsize=11.5)
        ax.set_ylabel(f"efficient coordinate  ({name})", fontsize=11.5)
        ax.set_title(f"{name}: efficient vs naive", fontsize=12.5, fontweight="bold")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.grid(alpha=0.25); ax.legend(fontsize=10)
    fig.suptitle("Naive and efficient Fastmap produce identical coordinates (points lie on y = x)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("parity_naive_vs_efficient.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved parity_naive_vs_efficient.png")


# ════════════════════════════════════════════════════════════════════════════
#  Figure 4 — Shepard diagram: distances preserved
# ════════════════════════════════════════════════════════════════════════════
def fig_shepard(D0, Xn, Xe):
    iu = np.triu_indices(D0.shape[0], k=1)
    orig = D0[iu]
    rec_n = pairwise_dist(Xn)[iu]
    rec_e = pairwise_dist(Xe)[iu]
    fig, ax = plt.subplots(figsize=(8.2, 7))
    lo, hi = orig.min(), orig.max()
    pad = 0.05 * (hi - lo)
    ax.plot([lo, hi], [lo, hi], ls="--", color=C_DARK, lw=1.8, label="perfect (y = x)")
    ax.scatter(orig, rec_n, s=30, color=C_ORANGE, alpha=0.7,
               edgecolor="white", linewidth=0.5, label=r"naive  $O(n^2 m)$", zorder=3)
    ax.scatter(orig, rec_e, s=34, color=C_BLUE, alpha=0.7, marker="^",
               edgecolor="white", linewidth=0.5, label=r"efficient  $O(nm^2)$", zorder=3)
    ax.set_xlabel("original pairwise distance  $d_0$", fontsize=12.5)
    ax.set_ylabel("reconstructed 2-D distance", fontsize=12.5)
    ax.set_title("Shepard diagram — both versions preserve distances equally well",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal"); ax.grid(alpha=0.25); ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("shepard_diagram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved shepard_diagram.png")


# ════════════════════════════════════════════════════════════════════════════
#  Figure 5 — empirical runtime scaling
# ════════════════════════════════════════════════════════════════════════════
def fig_runtime():
    ns = [200, 400, 800, 1600, 3200, 6400]
    m = 2
    t_naive, t_eff = [], []
    rng = np.random.default_rng(3)
    print("  (timing the core only; the O(n^2) input squaring is excluded)\n")
    for n in ns:
        P = rng.standard_normal((n, 4))
        D02 = pairwise_dist(P) ** 2            # squared input — precomputed, NOT timed
        _fastmap_naive_core(D02, m)            # warmup
        _fastmap_efficient_core(D02, m)
        reps_n = 5 if n <= 1600 else 3
        best_n = 1e9
        for _ in range(reps_n):                # naive core: the O(n^2 m) deflation
            t0 = time.perf_counter(); _fastmap_naive_core(D02, m)
            best_n = min(best_n, time.perf_counter() - t0)
        best_e = 1e9
        for _ in range(10):                    # efficient core: the O(n m^2) on-demand work
            t0 = time.perf_counter(); _fastmap_efficient_core(D02, m)
            best_e = min(best_e, time.perf_counter() - t0)
        t_naive.append(best_n); t_eff.append(best_e)
        print(f"  n={n:>5}   naive={best_n * 1e3:>9.3f} ms   "
              f"efficient={best_e * 1e3:>8.4f} ms   speedup={best_n / best_e:>9.1f}x")

    ns_arr = np.array(ns, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    ax = axes[0]
    ax.plot(ns_arr, t_naive, "o-", color=C_RED, lw=2.3, ms=8, label=r"naive  $O(n^2 m)$")
    ax.plot(ns_arr, t_eff, "s-", color=C_GREEN, lw=2.3, ms=8, label=r"efficient  $O(nm^2)$")
    ax.set_xlabel("number of objects  n", fontsize=12)
    ax.set_ylabel("wall-clock time (s)", fontsize=12)
    ax.set_title("Runtime vs n   (m = 2)", fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.25); ax.legend(fontsize=11)

    ax = axes[1]
    ax.loglog(ns_arr, t_naive, "o-", color=C_RED, lw=2.3, ms=8, label=r"naive  $O(n^2 m)$")
    ax.loglog(ns_arr, t_eff, "s-", color=C_GREEN, lw=2.3, ms=8, label=r"efficient  $O(nm^2)$")
    ax.loglog(ns_arr, t_naive[0] * (ns_arr / ns_arr[0]) ** 2, ":", color=C_RED,
              lw=1.5, alpha=0.7, label="slope-2 reference")
    ax.loglog(ns_arr, t_eff[0] * (ns_arr / ns_arr[0]), ":", color=C_GREEN,
              lw=1.5, alpha=0.7, label="slope-1 reference")
    ax.set_xlabel("number of objects  n", fontsize=12)
    ax.set_ylabel("wall-clock time (s)", fontsize=12)
    ax.set_title("log-log   (slope $\\approx 2$  vs  $\\approx 1$)", fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=9.5)

    fig.suptitle("Empirical complexity — efficient deflation removes the quadratic term",
                 fontsize=13.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig("runtime_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved runtime_scaling.png")
    return ns, t_naive, t_eff


# ════════════════════════════════════════════════════════════════════════════
#  Figure 6 — distance matrix shrinks under deflation
# ════════════════════════════════════════════════════════════════════════════
def fig_deflation_heatmaps(snapshots):
    titles = ["original distances  $d_0$",
              "after 1st-dim deflation",
              "after 2nd-dim deflation"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, M, ttl in zip(axes, snapshots[:3], titles):
        im = ax.imshow(M, cmap="viridis")
        ax.set_title(ttl, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Distance matrix shrinks as each pivot dimension is peeled off (deflation)",
                 fontsize=13.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig("deflation_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved deflation_heatmaps.png")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    banner("L9 — Fastmap efficient deflation  (target: O(n m^2))")

    # ── demo dataset: 3 blobs in 2-D ──
    true_pts, labels = make_blobs(n_per=20, centers=3, seed=7)
    D0 = pairwise_dist(true_pts)
    print(f"Demo: {true_pts.shape[0]} points, 3 blobs. Distance matrix {D0.shape}.\n")

    # ── run both implementations ──
    Xn, snaps = fastmap_naive(D0, m=2, want_snaps=True)
    Xe = fastmap_efficient(D0, m=2)

    banner("Correctness — max |naive - efficient| per coordinate")
    print(f"  dim 1 : {np.max(np.abs(Xn[:, 0] - Xe[:, 0])):.3e}")
    print(f"  dim 2 : {np.max(np.abs(Xn[:, 1] - Xe[:, 1])):.3e}\n")

    # ── figures ──
    fig_deflation_geometry()
    fig_embedding(true_pts, labels, Xe)
    fig_parity(Xn, Xe)
    fig_shepard(D0, Xn, Xe)
    fig_deflation_heatmaps(snaps)

    banner("Runtime scaling (m = 2)")
    ns, tn, te = fig_runtime()

    # ── final summary ──
    banner("RESULT")
    print(f"  efficient matches naive to ~{np.max(np.abs(Xn - Xe)):.1e} per coordinate")
    print(f"  speedup at n={ns[-1]}: ~{tn[-1] / te[-1]:.0f}x")
    print("=" * 74)


if __name__ == "__main__":
    main()
