import numpy as np
from scipy.special import softmax

# ======================================================
# 활성 expert 선정 / 마스크
# ======================================================
def get_active_token_mask_layer(router_logits_layer, topk_assign=8, prob_threshold=None):
    """
    router_logits_layer: (T, E)
    반환: (E, T) bool — 토큰별 상위 topk_assign (OR 확률 임계) 활성(True)
    """
    T, E = router_logits_layer.shape
    probs = softmax(router_logits_layer, axis=-1)  # (T, E)
    active = np.zeros((T, E), dtype=bool)

    if topk_assign is not None and topk_assign > 0:
        k = min(topk_assign, E)
        kth = np.argpartition(-probs, k-1, axis=1)[:, :k]  # (T,k)
        rows = np.repeat(np.arange(T), k)
        cols = kth.reshape(-1)
        active[rows, cols] = True
    if prob_threshold is not None:
        active |= (probs >= prob_threshold)

    return active.T  # (E, T)

def collect_activated_ids_in_layer(router_logits_layer, topk_assign=8, prob_threshold=None,
                                   min_activations=1, cap=None):
    """
    '활성 토큰 수 >= min_activations' expert를 활성 토큰 수 내림차순으로 정렬.
    cap 주어지면 상위 cap명만 반환.
    """
    act = get_active_token_mask_layer(router_logits_layer, topk_assign, prob_threshold)  # (E,T)
    counts_all = act.sum(axis=1)  # (E,)
    ids = np.where(counts_all >= min_activations)[0]
    counts = counts_all[ids]
    order = np.argsort(-counts)
    ids = ids[order].tolist()
    counts = counts[order].astype(int)
    if cap is not None and len(ids) > cap:
        ids = ids[:cap]
        counts = counts[:cap]
    return ids, counts

# ======================================================
# 레이어별 rep 수집 (+ 필요시 router 확률도 함께)
# ======================================================
def _collect_rep_sets_for_layer(
    expert_outputs_layer,   # (E, T, D)
    router_logits_layer,    # (T, E)
    *,
    n_experts=8,
    topk_assign=8,
    prob_threshold=None,
    min_activations=1,
    token_stride=1,
    pooling_weight="none"   # "none" | "router"
):
    stride = max(1, int(token_stride))
    T = expert_outputs_layer.shape[1]
    tok_idx = np.arange(T)[::stride]  # (Ts,)

    ids, _ = collect_activated_ids_in_layer(
        router_logits_layer, topk_assign=topk_assign, prob_threshold=prob_threshold,
        min_activations=min_activations, cap=n_experts
    )
    ids = list(ids)
    if len(ids) == 0:
        return [], [], None

    act_full = get_active_token_mask_layer(router_logits_layer, topk_assign, prob_threshold)  # (E,T)
    act = act_full[:, tok_idx]  # (E, Ts)

    use_router_w = (pooling_weight == "router")
    probs = softmax(router_logits_layer, axis=-1) if use_router_w else None

    rep_sets, expert_ids_used = [], []
    w_sets = [] if use_router_w else None

    for e in ids:
        m = act[e]
        idx = np.where(m)[0]
        X = expert_outputs_layer[e, tok_idx, :]
        X = X[idx, :]  # (m_e, D)
        if X.shape[0] < 1:
            continue

        rep_sets.append(X)
        expert_ids_used.append(e)

        if use_router_w:
            w = probs[tok_idx, e][idx].astype(np.float64)  # (m_e,)
            w_sets.append(w)

    return rep_sets, expert_ids_used, w_sets

# ======================================================
# 거리 유틸 (pairwise는 비가중, centroid는 가중 가능)
# ======================================================
def _avg_cross_l2(A, B):
    a2 = np.sum(A*A, axis=1, keepdims=True)
    b2 = np.sum(B*B, axis=1, keepdims=True).T
    ab = A @ B.T
    d2 = np.maximum(a2 + b2 - 2.0*ab, 0.0)
    d = np.sqrt(d2)
    return float(d.mean())

def _centroid_and_count(X, w=None, eps=1e-12):
    """
    X:(m,D) -> (mu:(D,), n_eff: float)
    - w=None  : 산술평균, n_eff = m
    - w!=None : 가중평균, n_eff = sum(w)
    """
    if (w is None) or (np.sum(w) <= eps):
        return X.mean(axis=0), float(X.shape[0])
    s = np.sum(w)
    mu = (w[:, None] * X).sum(axis=0) / s
    return mu, float(s)

def _l2_between_centroids(mu_i, mu_j):
    diff = mu_i - mu_j
    return float(np.sqrt(np.dot(diff, diff)))

# ======================================================
# Agglomerative (threshold) — pooling_weight은 centroid에서만 사용
# ======================================================
def agglomerative_pooling_threshold(
    rep_sets,                 # list[np.ndarray]
    *,
    threshold=0.5,
    distance_mode="pairwise", # "pairwise" | "centroid"
    pooling_weight="none",    # "none" | "router"   <-- NEW
    w_sets=None,              # list[np.ndarray] | None (내부용)
    max_merges=None,
    verbose=False,
    eps=1e-12
):
    if distance_mode not in ("pairwise", "centroid"):
        raise ValueError("distance_mode은 'pairwise' 또는 'centroid'만 지원합니다.")
    if pooling_weight not in ("none", "router"):
        raise ValueError("pooling_weight는 'none' 또는 'router'만 지원합니다.")

    clusters = [(i,) for i in range(len(rep_sets))]

    if distance_mode == "pairwise":
        # 비가중 모든 쌍 L2 평균
        rep_pooled = [rep_sets[i] for i in range(len(rep_sets))]
        def dist(i, j):
            return _avg_cross_l2(rep_pooled[i], rep_pooled[j])
        def merge_payload(i, j):
            return np.vstack([rep_pooled[i], rep_pooled[j]])

    else:
        # centroid 모드: pooling_weight에 따라 (가중)중심과 유효표본수 계산
        use_router_w = (pooling_weight == "router")
        if use_router_w:
            if w_sets is None:
                raise ValueError("pooling_weight='router'면 w_sets가 필요합니다.")
            centroids, counts = [], []
            for X, w in zip(rep_sets, w_sets):
                mu, n = _centroid_and_count(X, w)
                centroids.append(mu); counts.append(n)
        else:
            centroids, counts = [], []
            for X in rep_sets:
                mu, n = _centroid_and_count(X, None)
                centroids.append(mu); counts.append(n)

        def dist(i, j):
            return _l2_between_centroids(centroids[i], centroids[j])

        def merge_payload(i, j):
            n_i, n_j = counts[i], counts[j]
            mu_i, mu_j = centroids[i], centroids[j]
            n_new = n_i + n_j
            if n_new <= eps:
                mu_new = 0.5 * (mu_i + mu_j)
                n_new = eps
            else:
                mu_new = (n_i * mu_i + n_j * mu_j) / n_new
            return mu_new, n_new

    history = []
    if len(clusters) <= 1:
        if distance_mode == "pairwise":
            return clusters, rep_pooled, history
        else:
            rep_sizes = [int(n) for n in counts]
            return clusters, rep_sizes, history

    merges_done = 0
    while True:
        # 가장 가까운 쌍
        best_d, best_pair = None, None
        K = len(clusters)
        for i in range(K-1):
            for j in range(i+1, K):
                d = dist(i, j)
                if (best_d is None) or (d < best_d):
                    best_d, best_pair = d, (i, j)

        # 멈춤 조건
        if best_d is None or best_d > threshold:
            if verbose:
                print(f"[stop] best_d={best_d} > thr={threshold} (or None)")
            break

        # 병합
        i, j = best_pair
        if verbose:
            print(f"[merge] {clusters[i]} + {clusters[j]} @ d={best_d:.4f}")
        new_members = tuple(sorted(clusters[i] + clusters[j]))

        if distance_mode == "pairwise":
            X_new = merge_payload(i, j)
        else:
            mu_new, n_new = merge_payload(i, j)

        history.append({"pair": (clusters[i], clusters[j]),
                        "distance": float(best_d),
                        "new_cluster": new_members})

        # 상태 갱신
        ii, jj = max(i, j), min(i, j)
        clusters.pop(ii); clusters.pop(jj); clusters.append(new_members)

        if distance_mode == "pairwise":
            rep_pooled.pop(ii); rep_pooled.pop(jj); rep_pooled.append(X_new)
        else:
            centroids.pop(ii); counts.pop(ii)
            centroids.pop(jj); counts.pop(jj)
            centroids.append(mu_new); counts.append(n_new)

        merges_done += 1
        if (max_merges is not None) and (merges_done >= max_merges):
            if verbose:
                print(f"[stop] reached max_merges={max_merges}")
            break

    if distance_mode == "pairwise":
        rep_sizes = [X.shape[0] for X in rep_pooled]
        return clusters, rep_pooled, history
    else:
        rep_sizes = [int(n) for n in counts]
        return clusters, rep_sizes, history

# --- 정렬 유틸 그대로 ---
def _sort_clusters_stably(cluster_id_tuples):
    clusters = [tuple(sorted(cl)) for cl in cluster_id_tuples]
    clusters.sort(key=lambda c: (-len(c), c))
    return tuple(clusters)

# ======================================================
# 레이어 단위 wrapper
# ======================================================
def cluster_layer_by_pooling_threshold(
    expert_outputs_layer,   # (E, T, D)
    router_logits_layer,    # (T, E)
    *,
    n_experts=8,
    topk_assign=8,
    prob_threshold=None,
    min_activations=1,
    token_stride=1,
    threshold=0.5,
    distance_mode="pairwise",   # "pairwise" | "centroid"
    pooling_weight="none",      # "none" | "router"   <-- NEW
    verbose=False
):
    rep_sets, ids_used, w_sets = _collect_rep_sets_for_layer(
        expert_outputs_layer, router_logits_layer,
        n_experts=n_experts, topk_assign=topk_assign, prob_threshold=prob_threshold,
        min_activations=min_activations, token_stride=token_stride,
        pooling_weight=pooling_weight
    )
    if len(rep_sets) == 0:
        return {"clusters": [], "rep_sizes": [], "history": [], "ids_used": [], "distance_mode": distance_mode}
    if len(rep_sets) == 1:
        return {
            "clusters": [tuple([ids_used[0]])],
            "rep_sizes": [rep_sets[0].shape[0]],
            "history": [],
            "ids_used": ids_used,
            "distance_mode": distance_mode
        }

    clusters_idx, payload, history = agglomerative_pooling_threshold(
        rep_sets,
        threshold=threshold,
        distance_mode=distance_mode,
        pooling_weight=pooling_weight,
        w_sets=w_sets,
        verbose=verbose
    )

    clusters_ids = [tuple(ids_used[i] for i in cl) for cl in clusters_idx]
    clusters_ids = list(_sort_clusters_stably(clusters_ids))

    if distance_mode == "pairwise":
        rep_sizes = [rp.shape[0] for rp in payload]
    else:
        rep_sizes = payload  # sizes

    return {
        "clusters": tuple(clusters_ids),
        "rep_sizes": rep_sizes,
        "history": history,
        "ids_used": ids_used,
        "distance_mode": distance_mode
    }

# ======================================================
# 전체 레이어 실행
# ======================================================
def cluster_all_layers_by_pooling_threshold(
    expert_outputs, router_logits,
    *,
    n_experts=8,
    topk_assign=8,
    prob_threshold=None,
    min_activations=1,
    token_stride=1,
    threshold=0.5,
    distance_mode="pairwise",    # "pairwise" | "centroid"
    pooling_weight="none",       # "none" | "router"
    verbose=False
):
    L, E, T, D = expert_outputs.shape
    results = {}
    for l in range(L):
        res = cluster_layer_by_pooling_threshold(
            expert_outputs[l], router_logits[l],
            n_experts=n_experts, topk_assign=topk_assign, prob_threshold=prob_threshold,
            min_activations=min_activations, token_stride=token_stride,
            threshold=threshold, distance_mode=distance_mode,
            pooling_weight=pooling_weight, verbose=verbose
        )
        results[str(l)] = res
        if verbose:
            print(f"\n[Layer {l}] thr={threshold}, mode={distance_mode}, pool_w={pooling_weight}")
            for cl, sz in zip(res["clusters"], res["rep_sizes"]):
                print(f"  cluster {cl} -> reps={sz}")
    return results

# --- 간단 출력 ---
def cluster_all_layers_simple_output(
    expert_outputs, router_logits,
    *,
    n_experts=8,
    topk_assign=8,
    prob_threshold=None,
    min_activations=1,
    token_stride=1,
    threshold=0.5,
    distance_mode="pairwise",
    pooling_weight="none",
    verbose=False
):
    detailed = cluster_all_layers_by_pooling_threshold(
        expert_outputs, router_logits,
        n_experts=n_experts,
        topk_assign=topk_assign,
        prob_threshold=prob_threshold,
        min_activations=min_activations,
        token_stride=token_stride,
        threshold=threshold,
        distance_mode=distance_mode,
        pooling_weight=pooling_weight,
        verbose=verbose
    )
    return {l_str: res["clusters"] for l_str, res in detailed.items()}
