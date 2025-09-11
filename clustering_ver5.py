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
# 레이어별 활성 rep 수집 (초기 클러스터 = 각 expert의 활성 rep 집합)
# ======================================================
def _collect_rep_sets_for_layer(
    expert_outputs_layer,   # (E, T, D)
    router_logits_layer,    # (T, E)
    *,
    n_experts=8,
    topk_assign=8,
    prob_threshold=None,
    min_activations=1,
    token_stride=1
):
    """
    활성 상위 n_experts 선정 후, 각 expert의 '활성 토큰에서의 rep'만 모아 초기 집합을 만든다.
    반환:
      rep_sets: list[np.ndarray] 각 (m_e, D)
      expert_ids_used: list[int] rep_sets와 1:1 대응하는 expert id
    """
    stride = max(1, int(token_stride))
    T = expert_outputs_layer.shape[1]
    tok_idx = np.arange(T)[::stride]  # (Ts,)

    ids, _ = collect_activated_ids_in_layer(
        router_logits_layer, topk_assign=topk_assign, prob_threshold=prob_threshold,
        min_activations=min_activations, cap=n_experts
    )
    ids = list(ids)
    if len(ids) == 0:
        return [], []

    act_full = get_active_token_mask_layer(router_logits_layer, topk_assign, prob_threshold)  # (E,T)
    act = act_full[:, tok_idx]  # (E, Ts)

    rep_sets, expert_ids_used = [], []
    for e in ids:
        m = act[e]
        idx = np.where(m)[0]
        X = expert_outputs_layer[e, tok_idx, :]
        X = X[idx, :]  # (m_e, D)
        if X.shape[0] >= 1:
            rep_sets.append(X)
            expert_ids_used.append(e)

    return rep_sets, expert_ids_used

# ======================================================
# 거리 정의: (1) 모든 pair 평균 L2  (2) 중심(평균벡터) L2
# ======================================================
def _avg_cross_l2(A, B):
    """
    A:(m,D), B:(n,D) → 모든 쌍 L2 평균 (스칼라)
    """
    a2 = np.sum(A*A, axis=1, keepdims=True)      # (m,1)
    b2 = np.sum(B*B, axis=1, keepdims=True).T    # (1,n)
    ab = A @ B.T                                  # (m,n)
    d2 = np.maximum(a2 + b2 - 2.0*ab, 0.0)
    d = np.sqrt(d2)
    return float(d.mean())

def _centroid_and_count(X):
    """
    X:(m,D) → (mu:(D,), n:int)
    """
    return X.mean(axis=0), X.shape[0]

def _l2_between_centroids(mu_i, mu_j):
    diff = mu_i - mu_j
    return float(np.sqrt(np.dot(diff, diff)))

# ======================================================
# pooling-linkage agglomerative threshold 병합
#   distance_mode:
#     - "pairwise": 모든 pair 평균 L2 (정확 / 느림)
#     - "centroid": 중심(평균벡터) L2 (가중합/빠름)
# ======================================================
def agglomerative_pooling_threshold(
    rep_sets,                 # list[np.ndarray]
    *,
    threshold=0.5,            # 거리 임계값 (L2)
    distance_mode="pairwise", # "pairwise" | "centroid"
    max_merges=None,
    verbose=False
):
    if distance_mode not in ("pairwise", "centroid"):
        raise ValueError("distance_mode은 'pairwise' 또는 'centroid'만 지원합니다.")

    # 초기 클러스터 표현
    clusters = [(i,) for i in range(len(rep_sets))]

    if distance_mode == "pairwise":
        # 실제 rep를 쌓아 가며, 모든 pair L2 평균으로 거리 계산
        rep_pooled = [rep_sets[i] for i in range(len(rep_sets))]
        def dist(i, j):
            return _avg_cross_l2(rep_pooled[i], rep_pooled[j])
        def merge_payload(i, j):
            return np.vstack([rep_pooled[i], rep_pooled[j]])
    else:
        # centroid 모드: 각 클러스터를 (mu, n)로만 유지 → 빠르고 샘플 수로 자연스런 가중
        centroids = []
        counts = []
        for X in rep_sets:
            mu, n = _centroid_and_count(X)
            centroids.append(mu)
            counts.append(n)
        centroids = list(centroids)
        counts = list(counts)

        def dist(i, j):
            return _l2_between_centroids(centroids[i], centroids[j])

        def merge_payload(i, j):
            n_i, n_j = counts[i], counts[j]
            mu_i, mu_j = centroids[i], centroids[j]
            n_new = n_i + n_j
            mu_new = (n_i * mu_i + n_j * mu_j) / n_new
            return mu_new, n_new

    history = []
    if len(clusters) <= 1:
        if distance_mode == "pairwise":
            return clusters, rep_pooled, history
        else:
            # centroid 모드는 rep가 아닌 (mu,n)만 들고 있으므로, 표시용으로 count만 반환
            rep_sizes = counts[:]  # 크기 정보
            return clusters, rep_sizes, history

    merges_done = 0
    while True:
        # 1) 가장 가까운 쌍 찾기
        best_d, best_pair = None, None
        K = len(clusters)
        for i in range(K-1):
            for j in range(i+1, K):
                d = dist(i, j)
                if (best_d is None) or (d < best_d):
                    best_d, best_pair = d, (i, j)

        # 2) 멈춤 조건
        if best_d is None or best_d > threshold:
            if verbose:
                print(f"[stop] best_d={best_d} > thr={threshold} (or None)")
            break

        # 3) 병합 실행
        i, j = best_pair
        if verbose:
            print(f"[merge] pair={clusters[i]} + {clusters[j]} @ d={best_d:.4f}")
        new_members = tuple(sorted(clusters[i] + clusters[j]))

        if distance_mode == "pairwise":
            new_rep = merge_payload(i, j)  # vstack
        else:
            mu_new, n_new = merge_payload(i, j)

        history.append({
            "pair": (clusters[i], clusters[j]),
            "distance": float(best_d),
            "new_cluster": new_members
        })

        # 4) 상태 갱신 (큰 인덱스 먼저 pop)
        ii, jj = max(i, j), min(i, j)
        clusters.pop(ii)
        clusters.pop(jj)
        clusters.append(new_members)

        if distance_mode == "pairwise":
            rep_pooled.pop(ii)
            rep_pooled.pop(jj)
            rep_pooled.append(new_rep)
        else:
            centroids.pop(ii); counts.pop(ii)
            centroids.pop(jj); counts.pop(jj)
            centroids.append(mu_new); counts.append(n_new)

        merges_done += 1
        if (max_merges is not None) and (merges_done >= max_merges):
            if verbose:
                print(f"[stop] reached max_merges={max_merges}")
            break

    # 5) 반환 포맷 통일
    if distance_mode == "pairwise":
        # 각 클러스터의 rep 개수만 요약
        rep_sizes = [X.shape[0] for X in rep_pooled]
        return clusters, rep_pooled, history
    else:
        # centroid 모드에선 실제 rep는 안 들고 있으니, 사이즈 리스트만 반환
        rep_sizes = counts[:]
        return clusters, rep_sizes, history

# --- (A) 클러스터 정렬 유틸: 내부 오름차순, 바깥은 크기 내림차순/사전식 ---
def _sort_clusters_stably(cluster_id_tuples):
    # 내부 정렬
    clusters = [tuple(sorted(cl)) for cl in cluster_id_tuples]
    # 바깥 정렬: 길이 내림차순, 동일 길이는 사전식
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
    distance_mode="pairwise",  # "pairwise" | "centroid"
    verbose=False
):
    rep_sets, ids_used = _collect_rep_sets_for_layer(
        expert_outputs_layer, router_logits_layer,
        n_experts=n_experts, topk_assign=topk_assign, prob_threshold=prob_threshold,
        min_activations=min_activations, token_stride=token_stride
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
        rep_sets, threshold=threshold, distance_mode=distance_mode, verbose=verbose
    )

    # 출력 정리
    clusters_ids = [tuple(ids_used[i] for i in cl) for cl in clusters_idx]

    # ✅ 최종 정렬 적용
    clusters_ids = list(_sort_clusters_stably(clusters_ids))
    
    if distance_mode == "pairwise":
        rep_sizes = [rp.shape[0] for rp in payload]   # payload = rep_pooled
    else:
        rep_sizes = payload                            # payload = sizes

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
    distance_mode="pairwise",  # "pairwise" | "centroid"
    verbose=False
):
    """
    모든 레이어에 대해 pooling-linkage agglomerative threshold clustering 수행
    반환: dict[layer_str] = result(dict)
    """
    L, E, T, D = expert_outputs.shape
    results = {}
    for l in range(L):
        res = cluster_layer_by_pooling_threshold(
            expert_outputs[l], router_logits[l],
            n_experts=n_experts, topk_assign=topk_assign, prob_threshold=prob_threshold,
            min_activations=min_activations, token_stride=token_stride,
            threshold=threshold, distance_mode=distance_mode, verbose=verbose
        )
        results[str(l)] = res
        if verbose:
            print(f"\n[Layer {l}] thr={threshold}, mode={distance_mode}")
            for cl, sz in zip(res["clusters"], res["rep_sizes"]):
                print(f"  cluster {cl} -> reps={sz}")
    return results

# --- (C) 최종 간단 출력 래퍼: 원하는 형식으로 dict[str] -> tuple[tuple[int]] ---
def cluster_all_layers_simple_output(
    expert_outputs, router_logits,
    *,
    n_experts=8,
    topk_assign=8,
    prob_threshold=None,
    min_activations=1,
    token_stride=1,
    threshold=0.5,
    distance_mode="pairwise",  # "pairwise" | "centroid"
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
        verbose=verbose
    )
    simple = {}
    for l_str, res in detailed.items():
        # res["clusters"]는 이미 정렬된 tuple(tuple(...)) 형태
        simple[l_str] = res["clusters"]
    return simple
