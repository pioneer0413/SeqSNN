#!/usr/bin/env python3
"""
Metadata clustering and hierarchical dendrogram from a metadata dict or DataFrame.

- Accepts a Python dict like `data = { 'file_name': [...], 'freq': [...], ... }`
- Preprocess: encode categoricals, coerce numbers, boolean normalization
- Auto-select k via silhouette (fallback safe defaults)
- Produce: PCA cluster plot (SVG) and hierarchical dendrogram (SVG)
- Exposes analyze_metadata(data|df) for programmatic use, and a CLI-friendly main
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from itertools import combinations

# -----------------------------
# Utilities
# -----------------------------

def _to_bool_int_series(s: pd.Series) -> pd.Series:
    """Normalize boolean-like column to integers {False:0, True:1}.
    Handles True/False, 'TRUE'/'FALSE', 'yes'/'no', '1'/'0', etc.
    """
    if s.dtype == bool:
        return s.astype(int)
    mapping = {
        'TRUE': True, 'FALSE': False,
        'T': True, 'F': False,
        'YES': True, 'NO': False,
        'Y': True, 'N': False,
        '1': True, '0': False,
        1: True, 0: False,
        True: True, False: False,
    }
    s_norm = s.astype(str).str.upper().map(mapping)
    return s_norm.fillna(False).astype(int)


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')


def _encode_categoricals(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    encoders: Dict[str, LabelEncoder] = {}
    for c in cols:
        if c in df.columns:
            le = LabelEncoder()
            df[f"{c}_encoded"] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
    return df, encoders


# -----------------------------
# Core pipeline
# -----------------------------

def preprocess_metadata(data: Optional[Dict]=None, df: Optional[pd.DataFrame]=None) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Return (df_processed, X_scaled, feature_cols) ready for clustering.
    One of data (dict) or df must be provided.
    """
    if df is None:
        if data is None:
            raise ValueError("Provide either `data` dict or `df` DataFrame")
        df = pd.DataFrame(data)

    proc = df.copy()

    # Boolean-like columns (present in typical metadata)
    for col in ['if_univariate', 'trend', 'seasonal']:
        if col in proc.columns:
            proc[col] = _to_bool_int_series(proc[col])

    # Encode common categoricals
    proc, _ = _encode_categoricals(proc, ['freq', 'size'])

    # Coerce numerics
    numeric_cols = ['length', 'stationary', 'transition', 'shifting', 'correlation']
    _coerce_numeric(proc, numeric_cols)

    # Feature set (existence-checked)
    feature_cols_all = [
        'freq_encoded', 'size_encoded', 'if_univariate',
        'length', 'trend', 'seasonal', 'stationary', 'transition', 'shifting', 'correlation'
    ]
    feature_cols = [c for c in feature_cols_all if c in proc.columns]

    X = proc[feature_cols].copy()
    # Fill remaining NaNs per column with median (numeric) or 0 otherwise
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return proc, X_scaled, feature_cols


def choose_k(X: np.ndarray, max_k: int = 8) -> int:
    """Pick k by maximizing silhouette score over 2..max_k. Fallback to k=2.
    If samples < 3, fallback to k=2 (or 1 if really tiny).
    """
    n = X.shape[0]
    if n < 3:
        return 1 if n == 1 else 2
    best_k, best_score = 2, -1.0
    for k in range(2, min(max_k, n) + 1):
        try:
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score
        except Exception:
            continue
    return best_k


def run_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)


def plot_pca_clusters(X: np.ndarray, names: List[str], labels: np.ndarray, save_path: Optional[str] = None) -> None:
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sc = plt.scatter(Z[:, 0], Z[:, 1], c=labels, cmap='viridis', alpha=0.8)
    for i, name in enumerate(names):
        plt.annotate(str(name), (Z[i, 0], Z[i, 1]), fontsize=7, alpha=0.8)
    plt.title('KMeans Clusters (PCA)')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.colorbar(sc, label='Cluster ID')

    plt.subplot(1, 2, 2)
    comps = np.abs(pca.components_)
    idx = np.argsort(-comps[0])
    plt.barh(range(len(idx)), comps[0, idx], alpha=0.8)
    plt.gca().invert_yaxis()
    plt.yticks(range(len(idx)), idx)
    plt.title('PC1 | absolute component loadings (feature index)')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        fmt = ext[1:] if ext else 'svg'
        plt.savefig(save_path, bbox_inches='tight', format=fmt)
    plt.show()


def plot_dendrogram(
    X: np.ndarray,
    names: List[str],
    save_path: Optional[str] = None,
    method: str = 'ward',
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
) -> None:
    """Plot hierarchical dendrogram.

    You can highlight a grouping criterion by either specifying:
    - n_clusters: desired number of clusters (draws cut line at appropriate height)
    - distance_threshold: explicit distance cut height
    If both are None, the dendrogram is drawn without a cut line.
    """
    Z = linkage(X, method=method)

    feature_names = ['freq', 'size', 'if_univariate', 'length', 'trend', 'seasonal', 
                    'stationary', 'transition', 'shifting', 'correlation']
    trace = merge_trace_with_members(X, Z, names, feature_names=feature_names, topk=5, with_ward_attr=(method=='ward'))
    print_merge_trace(trace, max_lines_per_merge=10)

    # Determine cut height if n_clusters is provided
    cut_height = None
    if distance_threshold is not None:
        cut_height = float(distance_threshold)
    elif n_clusters is not None:
        n = len(names)
        if n_clusters <= 1:
            # Everything merged: choose above max height
            cut_height = float(np.max(Z[:, 2])) + 1e-9
        elif n_clusters >= n:
            # No merges: choose below first merge
            first = float(Z[0, 2]) if Z.shape[0] > 0 else 0.0
            cut_height = max(0.0, first - 1e-9)
        else:
            # Choose midpoint between the (m'-1)-th and m'-th merge distances
            # where m' = n - n_clusters merges are allowed
            d = Z[:, 2]
            mprime = n - n_clusters  # number of merges to allow
            # distances in Z are non-decreasing for standard methods (e.g., ward)
            # guard indexes
            lo = d[mprime - 1] if mprime - 1 >= 0 else d[0] - 1e-9
            hi = d[mprime] if mprime < len(d) else d[-1] + 1e-9
            cut_height = float((lo + hi) / 2.0)

    plt.figure(figsize=(12, 6))
    dendrogram(
        Z,
        labels=[str(nm) for nm in names],
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True,
        color_threshold=cut_height if cut_height is not None else None,
    )
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Dataset')
    plt.ylabel('Distance')
    plt.xticks(rotation=45, ha='right')
    # Draw the cut line if requested
    if cut_height is not None:
        plt.axhline(y=cut_height, color='crimson', linestyle='--', linewidth=1.5, label=f'cut @ {cut_height:.3g}')
        plt.legend(loc='upper right')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Infer format from extension; default to SVG if none
        ext = os.path.splitext(save_path)[1].lower()
        fmt = ext[1:] if ext else 'svg'
        plt.savefig(save_path, bbox_inches='tight', format=fmt)
    plt.show()

def merge_trace_with_members(X, Z, names, feature_names=None, topk=5, with_ward_attr=False, contrib_type='relative'):
    """
    Track members per merge and (optionally) Ward per-feature contributions.
    Returns a list of dicts, one per merge in Z.
    """
    n, d = X.shape
    
    if feature_names is None:
        feature_names = [f"f{j}" for j in range(d)]  # fallback
    
    # initialize leaf clusters
    members = {i: [i] for i in range(n)}          # id -> list of indices
    sizes   = {i: 1 for i in range(n)}
    means   = {i: X[i].astype(float) for i in range(n)}  # for Ward only

    results = []
    for t, (a, b, dist, size) in enumerate(Z):
        a, b = int(a), int(b)
        new_id = n + t

        # --- member tracking (method-agnostic) ---
        memb_a = members[a]
        memb_b = members[b]
        memb_new = memb_a + memb_b

        entry = {
            "merge_index": t,
            "clusters_merged": (a, b),
            "new_cluster": new_id,
            "distance": float(dist),
            "size_a": sizes[a],
            "size_b": sizes[b],
            "new_size": len(memb_new),
            "members_a": [(i, names[i]) for i in sorted(memb_a)],
            "members_b": [(i, names[i]) for i in sorted(memb_b)],
            "members_new": [(i, names[i]) for i in sorted(memb_new)],
        }

        # --- optional: Ward per-feature contribution (exact) ---
        if with_ward_attr:
            na, nb = sizes[a], sizes[b]
            mu_a, mu_b = means[a], means[b]
            coef = (na * nb) / (na + nb)
            per_feat = coef * (mu_a - mu_b) ** 2  # shape (d,)
            
            if contrib_type == 'relative':
                total = np.sum(per_feat)
                if total > 0:
                    per_feat = per_feat / total
                else:
                    per_feat = np.zeros_like(per_feat)

            # sort and map indices -> names
            idx = np.argsort(per_feat)[::-1]
            top_idx = idx[:topk]

            entry["top_features"] = [
                (feature_names[int(j)], float(per_feat[int(j)])) for j in top_idx
            ]
            # Optional: full name->value mapping (can be large)

            entry["per_feature_contrib"] = per_feat  # numeric vector
            entry["per_feature_contrib_named"] = {
                feature_names[j]: float(per_feat[j]) for j in range(d)
            }

        results.append(entry)

        # update maps
        members[new_id] = memb_new
        sizes[new_id] = len(memb_new)
        if with_ward_attr:
            # compute mean BEFORE removing a/b
            new_size = sizes[a] + sizes[b]
            means[new_id] = (sizes[a] * means[a] + sizes[b] * means[b]) / new_size

        # free old ids
        for k in (a, b):
            members.pop(k, None)
            sizes.pop(k, None)
            if with_ward_attr:
                means.pop(k, None)

    return results

def print_merge_trace(
    results,
    max_lines_per_merge=3,
    sort_key="new_size",      # "new_size", "distance", or "merge_index"
    ascending=True
):
    # choose key function
    def key_fn(r):
        primary = r.get(sort_key, float("inf"))
        # tie-breakers: distance, then merge_index
        return (
            primary,
            r.get("distance", float("inf")),
            r.get("merge_index", float("inf")),
        )

    # sort results
    sorted_results = sorted(results, key=key_fn, reverse=not ascending)

    for r in sorted_results:
        print(
            f"merge {r['merge_index']}: cluster {r['clusters_merged'][0]} "
            f"+ cluster {r['clusters_merged'][1]} -> new cluster {r['new_cluster']}, "
            f"distance={r['distance']:.6g}, size={r['new_size']}"
        )

        def fmt(lst):
            items = ", ".join(f"{i}:{nm}" for i, nm in lst[:max_lines_per_merge])
            more = "" if len(lst) <= max_lines_per_merge else f", ...(+{len(lst)-max_lines_per_merge})"
            return items + more

        print("  members_a:", fmt(r["members_a"]))
        print("  members_b:", fmt(r["members_b"]))
        print("  members_new:", fmt(r["members_new"]))
        if "top_features" in r:
            print("  top_features:", r["top_features"])

def _match_merges_to_plot_segments(Z, ddata, tol=1e-12):
    """
    Map each merge (row in Z) to a plotted segment index in dendrogram output.
    Returns a list 'seg_idx_for_merge' of length Z.shape[0], where each entry
    is the index k into ddata['icoord']/['dcoord'] corresponding to that merge.

    Strategy:
    - For each plotted segment k, take its height hk = max(dcoord[k]).
    - For each merge i with height hi = Z[i,2], match to an unused k with |hk-hi| minimal.
    - Resolve ties by increasing height, then by index.
    """
    heights_Z = Z[:, 2].astype(float)
    seg_heights = np.array([max(dc) for dc in ddata["dcoord"]], dtype=float)

    # sort merges and segments by height, then greedy match
    merge_order = np.argsort(heights_Z)
    seg_order = np.argsort(seg_heights)

    used_seg = np.zeros(len(seg_heights), dtype=bool)
    seg_idx_for_merge = [-1] * len(heights_Z)

    j = 0
    for i in merge_order:
        hi = heights_Z[i]
        # advance seg pointer j until seg_height >= hi - tol
        while j < len(seg_order) and seg_heights[seg_order[j]] < hi - tol:
            j += 1
        # candidate window around j
        candidates = []
        if j < len(seg_order):
            candidates.append(seg_order[j])
        if j - 1 >= 0:
            candidates.append(seg_order[j - 1])
        # add a small neighborhood to be safe
        for off in (1, 2):
            if j + off < len(seg_order):
                candidates.append(seg_order[j + off])
            if j - 1 - off >= 0:
                candidates.append(seg_order[j - 1 - off])
        # pick unused candidate with smallest |hk-hi|
        best_k, best_diff = -1, float("inf")
        for k in candidates:
            if 0 <= k < len(seg_heights) and not used_seg[k]:
                diff = abs(seg_heights[k] - hi)
                if diff < best_diff:
                    best_k, best_diff = k, diff
        # fallback: global nearest unused
        if best_k == -1:
            for k in np.argsort(np.abs(seg_heights - hi)):
                if not used_seg[k]:
                    best_k = k
                    best_diff = abs(seg_heights[k] - hi)
                    break
        used_seg[best_k] = True
        seg_idx_for_merge[i] = best_k

    return seg_idx_for_merge  # list of indices into ddata["icoord"]/["dcoord"]


def plot_dendrogram_with_annotations(
    X: np.ndarray,
    names: List[str],
    feature_names: Optional[List[str]] = None,
    method: str = "ward",
    topk: int = 5,
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    annotate_all: bool = True,
    min_height: Optional[float] = None,
    save_path: Optional[str] = None,
):
    """
    Plot a hierarchical dendrogram and (for Ward) annotate each merge with top-k feature contributors.

    Parameters
    ----------
    X : array (n_samples, n_features)
    names : list of leaf labels (length n_samples)
    feature_names : list of feature names (length n_features). If None, uses f0..f{d-1}.
    method : str, one of scipy linkage methods. For non-Ward, feature attributions are skipped.
    topk : int, number of top features to display per merge (Ward only)
    n_clusters : optional int, draw cut line by number of clusters
    distance_threshold : optional float, draw cut line at this height
    annotate_all : bool, if False only annotate top ~25% highest merges to reduce clutter
    min_height : optional float, annotate only merges with height >= min_height
    save_path : optional str, save figure if provided (format inferred from extension)
    """
    # compute linkage
    Z = linkage(X, method=method)

    '''
    explain_ward_merge_choices(
        X, Z, names,
        feature_names=feature_names,
        steps=[0],               # analyze step 0
        topk_features=3,
        list_top_candidates=5,
        show_relative=True
    )'''

    # compute cut height if any
    cut_height = None
    if distance_threshold is not None:
        cut_height = float(distance_threshold)
    elif n_clusters is not None:
        n = len(names)
        if n_clusters <= 1:
            cut_height = float(np.max(Z[:, 2])) + 1e-9
        elif n_clusters >= n:
            first = float(Z[0, 2]) if Z.shape[0] > 0 else 0.0
            cut_height = max(0.0, first - 1e-9)
        else:
            d = Z[:, 2]
            mprime = n - n_clusters
            lo = d[mprime - 1] if mprime - 1 >= 0 else d[0] - 1e-9
            hi = d[mprime] if mprime < len(d) else d[-1] + 1e-9
            cut_height = float((lo + hi) / 2.0)

    # draw (and capture) dendrogram structure for annotation
    plt.figure(figsize=(15, 12))
    ddata = dendrogram(
        Z,
        labels=[str(nm) for nm in names],
        orientation="top",
        distance_sort="descending",
        show_leaf_counts=True,
        color_threshold=cut_height if cut_height is not None else None,
        no_plot=False,   # we want to plot and also capture coords
    )
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Dataset")
    plt.ylabel("Distance")
    plt.xticks(rotation=45, ha="right")

    # draw cut line if requested
    if cut_height is not None:
        plt.axhline(y=cut_height, linestyle="--", linewidth=1.5, label=f"cut @ {cut_height:.3g}")
        plt.legend(loc="upper right")

    # prepare annotations for Ward
    do_attr = (method.lower() == "ward")
    if do_attr:
        # feature names fallback
        d = X.shape[1]
        if feature_names is None:
            feature_names = [f"f{j}" for j in range(d)]

        # compute per-merge members + ward attributions
        trace = merge_trace_with_members(
            X, Z, names,
            feature_names=feature_names,
            topk=topk,
            with_ward_attr=True,
            contrib_type='relative',
        )

        # map merges to plotted segments
        seg_idx_for_merge = _match_merges_to_plot_segments(Z, ddata)

        # decide which merges to annotate (to avoid clutter)
        heights = Z[:, 2]
        annotate_mask = np.ones_like(heights, dtype=bool)
        if not annotate_all:
            # keep top ~25% highest merges
            thresh = np.quantile(heights, 0.75)
            annotate_mask = heights >= thresh
        if min_height is not None:
            annotate_mask &= heights >= float(min_height)

        # place annotations
        for i, r in enumerate(trace):
            if not annotate_mask[i]:
                continue
            k = seg_idx_for_merge[i]
            # the "V" segment: put text at the midpoint of the two inner x's at height h
            xs = ddata["icoord"][k]
            ys = ddata["dcoord"][k]
            h = max(ys)
            # inner branches are located at indices 1 and 2 in the icoord list
            xm = 0.5 * (xs[1] + xs[2])
            ym = h

            # build label: "f1:0.123, f2:0.045, f3:0.010"
            if "top_features" in r and len(r["top_features"]) > 0:
                parts = []
                for name, val in r["top_features"][:topk]:
                    parts.append(f"{name}: {val:.2f}")
                label = "\n".join(parts)
                plt.text(xm, ym, label, ha="center", va="bottom", rotation=0, fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        fmt = ext[1:] if ext else "svg"
        plt.savefig(save_path, bbox_inches="tight", format=fmt)
    plt.show()

def _ward_pair_cost(mu_a, mu_b, n_a, n_b):
    """Return (per_feature_contrib, total) for a candidate Ward merge."""
    coef = (n_a * n_b) / (n_a + n_b)
    per_feat = coef * (mu_a - mu_b) ** 2
    total = float(per_feat.sum())
    return per_feat, total

def explain_ward_merge_choices(
    X,
    Z,
    names,
    feature_names=None,
    steps=None,                 # e.g., [0] or [0,1,2]; None = all steps
    topk_features=3,            # top-k features to display for the chosen pair
    list_top_candidates=5,      # how many smallest-cost candidate pairs to print
    show_relative=True,         # also show relative (%) feature contributions
    warn_large=True,            # warn if candidate pairs are huge
):
    """
    For each agglomerative step t (before performing merge t in Z), compute all
    candidate Ward merge costs among active clusters and print a table sorted by ΔSSE.
    Also show top-k feature contributors for the actually selected pair.

    Assumes method='ward' was used to produce Z.
    """
    n, d = X.shape
    if feature_names is None:
        feature_names = [f"f{j}" for j in range(d)]
    elif len(feature_names) != d:
        raise ValueError("feature_names length must match X.shape[1].")

    # Maintain active clusters: ids 0..n-1 initially; new ids n..n+(n-2)
    active_means = {i: X[i].astype(float) for i in range(n)}
    active_sizes = {i: 1 for i in range(n)}

    # Decide which steps to analyze
    all_steps = range(Z.shape[0])
    if steps is None:
        steps = list(all_steps)
    else:
        steps = [int(t) for t in steps if 0 <= t < Z.shape[0]]

    for t in all_steps:
        a, b, dist, size_new = map(int, (Z[t,0], Z[t,1], Z[t,2], Z[t,3]))

        # ---- Before executing merge t, optionally analyze candidates ----
        if t in steps:
            active_ids = sorted(active_means.keys())
            m = len(active_ids)
            num_pairs = m * (m - 1) // 2
            if warn_large and num_pairs > 50000:
                print(f"[step {t}] Warning: {num_pairs} candidate pairs (m={m}). "
                      f"Computation may be heavy.")

            rows = []
            # compute all candidate costs
            for i, j in combinations(active_ids, 2):
                mu_i, mu_j = active_means[i], active_means[j]
                ni, nj = active_sizes[i], active_sizes[j]
                per_feat, total = _ward_pair_cost(mu_i, mu_j, ni, nj)
                rows.append({
                    "pair": (i, j),
                    "total": total,
                    "per_feat": per_feat,
                })

            # sort by total ascending
            rows.sort(key=lambda r: r["total"])

            print(f"\n=== step {t}: active={m} clusters; candidate pairs={num_pairs} ===")
            print(f"chosen merge in Z: ({a}, {b}) -> new id {n+t}  |  height={dist:.6g}  |  new_size={size_new}")

            # print top-k cheapest candidates (including the chosen one even if not in top)
            printed = 0
            chosen_shown = False
            for r in rows:
                i, j = r["pair"]
                flag = "*" if ((i==a and j==b) or (i==b and j==a)) else " "
                if flag == "*":
                    chosen_shown = True
                if printed < list_top_candidates or flag == "*":
                    print(f"{flag}  pair ({i},{j})  ΔSSE={r['total']:.6g}")
                    printed += 1

            if not chosen_shown:
                # print the chosen pair separately if not in the top list
                r = next(rr for rr in rows
                         if (rr["pair"]==(a,b) or rr["pair"]==(b,a)))
                print(f"*  pair ({a},{b})  ΔSSE={r['total']:.6g}  (chosen)")

            # show feature top-k for the chosen pair
            chosen = next(rr for rr in rows
                          if (rr["pair"]==(a,b) or rr["pair"]==(b,a)))
            per_feat = chosen["per_feat"]
            total = per_feat.sum()
            order = np.argsort(per_feat)[::-1]
            print("  top features (absolute):")
            for j in order[:topk_features]:
                print(f"    {feature_names[int(j)]:>16}: {per_feat[int(j)]:>10.3g}")
            if show_relative and total > 0:
                rel = per_feat / total
                print("  top features (relative %):")
                for j in order[:topk_features]:
                    print(f"    {feature_names[int(j)]:>16}: {rel[int(j)]:>9.2%}")

        # ---- Execute merge t to update active clusters (so next step is correct) ----
        # means & sizes BEFORE removing a,b
        mu_a = active_means[a]; mu_b = active_means[b]
        na   = active_sizes[a]; nb   = active_sizes[b]
        new_id = n + t
        active_means[new_id] = (na * mu_a + nb * mu_b) / (na + nb)
        active_sizes[new_id] = na + nb
        # remove merged
        active_means.pop(a, None); active_means.pop(b, None)
        active_sizes.pop(a, None); active_sizes.pop(b, None)

# -----------------------------
# Public API
# -----------------------------

def summarize_clusters(
    df_proc: pd.DataFrame,
    feature_cols: List[str],
    labels: np.ndarray,
    top_n: int = 5,
    save_dir: Optional[str] = None,
) -> Dict:
    """Produce per-cluster summaries of key features.

    - Numeric features: top-N by absolute z-score vs overall mean
    - Boolean-like (0/1): proportion of 1s
    - Encoded categoricals: mean of encoded values (proxy) and overall mean diff
    Saves:
      - cluster_summary.txt (human-readable)
      - cluster_stats.csv (means per cluster)
      - cluster_counts.csv (counts per cluster)
    Returns a dict with basic tables for programmatic use.
    """
    out: Dict[str, object] = {}
    data = df_proc.copy()
    data['cluster'] = labels

    # Basic tables
    counts = data.groupby('cluster').size().rename('count')
    means = data.groupby('cluster')[feature_cols].mean(numeric_only=True)
    overall_mean = data[feature_cols].mean(numeric_only=True)

    # Compute z-score-like deviation per cluster per feature
    std = data[feature_cols].std(numeric_only=True).replace(0, np.nan)
    zscores = (means - overall_mean) / std

    # Build human-readable highlights
    lines: List[str] = []
    for c in sorted(counts.index):
        lines.append(f"Cluster {c} | n={counts.loc[c]}")
        if c in zscores.index:
            zs = zscores.loc[c].dropna().abs().sort_values(ascending=False)
            top_feats = zs.head(top_n).index.tolist()
            lines.append("  Top deviations: " + ", ".join(top_feats))
        # Show selected boolean/categorical proportions if present
        for bcol in ['if_univariate', 'trend', 'seasonal']:
            if bcol in feature_cols and bcol in data.columns:
                prop = data.loc[data['cluster'] == c, bcol].mean()
                lines.append(f"  {bcol}=1 proportion: {prop:.2f}")
        lines.append("")

    # Save outputs
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        txt_path = os.path.join(save_dir, 'cluster_summary.txt')
        with open(txt_path, 'w') as f:
            f.write("\n".join(lines))
        means_path = os.path.join(save_dir, 'cluster_stats.csv')
        counts_path = os.path.join(save_dir, 'cluster_counts.csv')
        means.to_csv(means_path)
        counts.to_csv(counts_path)
        out['summary_txt'] = txt_path
        out['means_csv'] = means_path
        out['counts_csv'] = counts_path

    out['counts'] = counts
    out['means'] = means
    out['top_n'] = top_n
    return out

def analyze_metadata(
    data: Optional[Dict] = None,
    df: Optional[pd.DataFrame] = None,
    save_dir: str = '/home/hwkang/SeqSNN/outputs',
    max_k: int = 8,
    dendrogram_n_clusters: Optional[int] = None,
    dendrogram_distance_threshold: Optional[float] = None,
) -> Dict:
    """Full pipeline: preprocess -> choose k -> kmeans -> plots (PCA + dendrogram).

    Returns a dict with: df_processed, k, labels (np.ndarray), feature_cols, paths
    """
    proc, X, feature_cols = preprocess_metadata(data=data, df=df)

    names = proc['file_name'].astype(str).str.replace('.csv', '', regex=False).tolist() if 'file_name' in proc.columns else [str(i) for i in range(len(proc))]

    k = choose_k(X, max_k=max_k)
    labels = run_kmeans(X, k)

    paths = {}
    pca_path = os.path.join(save_dir, 'metadata_clusters_pca.svg')
    dendro_path = os.path.join(save_dir, 'metadata_dendrogram.svg')

    plot_pca_clusters(X, names, labels, save_path=pca_path)
    # Show grouping criterion: prefer user-provided values, else chosen k
    cut_k = dendrogram_n_clusters if dendrogram_n_clusters is not None else k

    plot_dendrogram_with_annotations(
        X,
        names,
        feature_names=feature_cols,
        method='ward',
        topk=3,
        n_clusters=cut_k,
        distance_threshold=dendrogram_distance_threshold,
        annotate_all=True,
        min_height=None,
        save_path=os.path.join(save_dir, 'metadata_dendrogram_annotated.svg'),
    )

    # Summarize cluster-wise key characteristics
    summary = summarize_clusters(
        proc,
        feature_cols,
        labels,
        top_n=5,
        save_dir=save_dir,
    )

    return {
        'df_processed': proc,
        'k': k,
        'labels': labels,
        'feature_cols': feature_cols,
        'paths': {
            'pca_svg': pca_path,
            'dendrogram_svg': dendro_path,
            # Decision Tree outputs may be added below if present
        },
        'cluster_summary': summary,
    }


if __name__ == '__main__':
    data = {
            'file_name': ['Covid-19.csv', 'CzeLan.csv', 'Electricity.csv', 'ETTh1.csv', 'ETTh2.csv', 
                            'ETTm1.csv', 'ETTm2.csv', 'Exchange.csv', 'FRED-MD.csv', 'METR-LA.csv',
                            'NASDAQ.csv', 'ILI.csv', 'NN5.csv', 'NYSE.csv', 'PEMS04.csv',
                            'PEMS08.csv', 'PEMS-BAY.csv', 'AQShunyi.csv', 'AQWan.csv', 'Solar.csv',
                            'Traffic.csv', 'Weather.csv', 'Wike2000.csv', 'Wind.csv', 'ZafNoo.csv'],
            'freq': ['daily', 'mins', 'hourly', 'hourly', 'hourly', 'mins', 'mins', 'daily', 
                    'monthly', 'mins', 'daily', 'weekly', 'daily', 'daily', 'mins',
                    'mins', 'mins', 'hourly', 'hourly', 'mins', 'hourly', 'mins', 
                    'daily', 'mins', 'mins'],
            'if_univariate': [False] * 25,
            'size': ['large'] * 25,
            'length': [1392, 19934, 26304, 14400, 14400, 57600, 57600, 7588, 728, 34272,
                        1244, 966, 791, 1243, 16992, 17856, 52116, 35064, 35064, 52560,
                        17544, 52696, 792, 48673, 19225],
            'trend': [True, False, False, True, True, False, True, True, True, False,
                        True, False, False, True, False, False, False, False, False, False,
                        False, False, False, False, False],
            'seasonal': [False, False, True, False, False, False, False, False, False, False,
                        False, False, False, False, False, False, False, False, False, True,
                        False, False, False, False, False],
            'stationary': [0.322533571, 0.159632874, 0.005149945, 0.001199963, 0.02178793, 
                            9.73e-05, 0.003043099, 0.359773614, 0.573526338, 2.48e-17,
                            0.169256065, 0.169154899, 0.028982465, 0.679431911, 7.18e-26,
                            7.56e-15, 7.55e-28, 0.000301746, 0.000274926, 0, 3.71e-08,
                            1.04e-08, 0.078095443, 0.007370192, 0.043392201],
            'transition': [0.125898016, 0.053166265, 0.010486153, 0.019838082, 0.042026418,
                            0.026867109, 0.037653009, 0.062335128, 0.114277636, 0.005812772,
                            0.074074074, 0.037777378, 0.006903295, 0.166666667, 0.006927345,
                            0.005458237, 0.004930629, 0.032323626, 0.032040059, 0.02524516,
                            0.01087732, 0.03678061, 0.037111414, 0.028041159, 0.034443575],
            'shifting': [0.23627084, -0.158231254, -0.07494365, -0.061362145, -0.403813351,
                        -0.062976669, -0.405554986, 0.325340999, 0.394262863, 0.005596618,
                        0.931752412, 0.721088435, 0.195219872, -0.61995173, 0.065983916,
                        0.024812223, 0.061538154, 0.018716425, -0.011412897, 0.198114051,
                        0.066992351, 0.213569048, -0.103750316, 0.132267523, -0.078172361],
            'correlation': [0.604030075, 0.683253051, 0.802452708, 0.63015285, 0.509002938,
                            0.612411309, 0.503586222, 0.565480222, 0.659990217, 0.769894181,
                            0.563610902, 0.67422445, 0.713436141, 0.612897825, 0.796510895,
                            0.806689813, 0.815379397, 0.612853502, 0.624402773, 0.785124345,
                            0.813523957, 0.694154841, 0.718651695, 0.507246107, 0.598541943]
        }

    result = analyze_metadata(data=data)

    print(f"Chosen k: {result['k']}")
    print(f"Saved: {result['paths']}")