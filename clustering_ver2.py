import numpy as np
from scipy.special import softmax
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
import matplotlib.pyplot as plt

# ============ 공통 유틸 ============
def _center_rows(X):
    return X - X.mean(axis=0, keepdims=True)

def get_active_token_mask_layer(router_logits_layer, topk_assign=8, prob_threshold=None):
    """
    router_logits_layer: (T, E)
    반환: (E, T) bool — 토큰별 상위 topk_assign (OR 확률 임계)을 활성(True)로 표기
    """
    T, E = router_logits_layer.shape
    probs = softmax(router_logits_layer, axis=-1)  # (T, E)
    active = np.zeros((T, E), dtype=bool)

    if topk_assign is not None and topk_assign > 0:
        k = min(topk_assign, E)
        kth = np.argpartition(-probs, k-1, axis=1)[:, :k]
        rows = np.repeat(np.arange(T), k)
        cols = kth.reshape(-1)
        active[rows, cols] = True
    if prob_threshold is not None:
        active |= (probs >= prob_threshold)

    return active.T  # (E, T)

def collect_activated_ids_in_layer(router_logits_layer, topk_assign=8, prob_threshold=None,
                                   min_activations=1):
    """
    해당 레이어에서 '활성 토큰 수 >= min_activations' 인 expert ID 목록과 활성 카운트 반환
    """
    act = get_active_token_mask_layer(router_logits_layer, topk_assign, prob_threshold)  # (E,T)
    counts = act.sum(axis=1)  # (E,)
    ids = np.where(counts >= min_activations)[0]
    counts = counts[ids]
    order = np.argsort(-counts)
    return ids[order].tolist(), counts[order].astype(int)

# ============ CKA (linear) ============
def compute_linear_cka(X, Y, sample_weights=None, eps=1e-12):
    """
    X: (N, Dx), Y: (N, Dy) — 샘플축: 토큰들
    sample_weights: (N,) optional — sqrt(w) 행 스케일링(합을 N으로 정규화)
    """
    if sample_weights is not None:
        w = np.asarray(sample_weights).reshape(-1, 1)
        w = w * (X.shape[0] / (np.sum(w) + eps))
        sw = np.sqrt(np.maximum(w, 0.0))
        X = X * sw
        Y = Y * sw
    Xc = _center_rows(X)
    Yc = _center_rows(Y)
    XtY = Xc.T @ Yc
    num = np.linalg.norm(XtY, 'fro') ** 2
    denom = (np.linalg.norm(Xc.T @ Xc, 'fro') * np.linalg.norm(Yc.T @ Yc, 'fro')) + eps
    return float(np.clip(num / denom, 0.0, 1.0))

# ============ 레이어 단위: 활성-only 쌍별 CKA ============
def pairwise_cka_condensed_layer_activated(
    expert_outputs_layer,   # (E, T, D)
    router_logits_layer,    # (T, E)
    expert_ids,             # list[int], 길이 k
    *,
    topk_assign=8,
    prob_threshold=None,
    min_samples=8,
    pair_policy='max',      # 'max' -> 교집합 < min_samples 이면 거리=1.0, 'nan' -> np.nan
    weight_by_router=False,
    token_stride=1
):
    """
    반환: condensed distances (1-CKA), pair_counts  (길이 k*(k-1)/2)
    거리 계산은 '두 expert가 동시에 활성'인 토큰 교집합에서만 수행.
    """
    E, T, D = expert_outputs_layer.shape
    tok_idx = np.arange(T)[::max(1, token_stride)]
    # 활성 마스크
    act = get_active_token_mask_layer(router_logits_layer, topk_assign, prob_threshold)  # (E,T)
    act = act[:, tok_idx]  # (E, Ts)

    # (선택) 라우터 가중
    probs = softmax(router_logits_layer, axis=-1) if weight_by_router else None
    mats = [expert_outputs_layer[e, tok_idx, :] for e in expert_ids]  # [(Ts, D)]
    wts  = [probs[tok_idx, e] if weight_by_router else None for e in expert_ids]

    k = len(expert_ids)
    dists, counts = [], []
    for i in range(k-1):
        for j in range(i+1, k):
            mask = act[expert_ids[i]] & act[expert_ids[j]]
            idx  = np.where(mask)[0]
            if idx.size >= min_samples:
                Xi, Xj = mats[i][idx], mats[j][idx]
                w = None
                if weight_by_router and (wts[i] is not None) and (wts[j] is not None):
                    w = (wts[i][idx] + wts[j][idx]) / 2.0
                cka = compute_linear_cka(Xi, Xj, w)
                dists.append(1.0 - cka); counts.append(idx.size)
            else:
                if pair_policy == 'max':
                    dists.append(1.0); counts.append(idx.size)
                else:
                    dists.append(np.nan); counts.append(idx.size)
    return np.asarray(dists, float), np.asarray(counts, int)

# ============ 레이어별 클러스터링(8개로 컷) ============
def cluster_one_layer_activated(
    expert_outputs_layer,   # (E, T, D)
    router_logits_layer,    # (T, E)
    *,
    n_clusters=8,
    topk_assign=8,
    prob_threshold=None,
    min_activations=1,
    min_samples=8,
    pair_policy='max',
    weight_by_router=False,
    token_stride=1,
    plot=False,
    layer_idx=None
):
    """
    해당 레이어에서 활성된 expert들만 대상으로,
    활성 교집합 기반 CKA 거리로 계층적 군집(average linkage) 후
    클러스터를 '정확히 n_clusters'로 잘라 그룹을 반환.
    """
    expert_ids, counts = collect_activated_ids_in_layer(
        router_logits_layer, topk_assign, prob_threshold, min_activations
    )
    k = len(expert_ids)
    if k < 2:
        return {"clusters": tuple(), "ids": tuple(), "counts": np.asarray(counts), "order": tuple(), "linkage": None}

    # 쌍별 거리
    condensed, pair_counts = pairwise_cka_condensed_layer_activated(
        expert_outputs_layer, router_logits_layer, expert_ids,
        topk_assign=topk_assign, prob_threshold=prob_threshold,
        min_samples=min_samples, pair_policy=pair_policy,
        weight_by_router=weight_by_router, token_stride=token_stride
    )
    if np.isnan(condensed).any():
        raise ValueError("NaN distance 발생: pair_policy='max' 사용 또는 활성 기준을 완화하세요.")

    # average linkage
    Z = linkage(condensed, method='average')
    order = leaves_list(Z)

    # n_clusters로 컷
    n_use = min(n_clusters, k)
    labels = fcluster(Z, t=n_use, criterion='maxclust')  # 1..n_use
    # 클러스터를 expert ID로 묶어 튜플 형태로 정렬
    clusters = []
    for c in range(1, n_use + 1):
        members = [expert_ids[i] for i in range(k) if labels[i] == c]
        clusters.append(tuple(members))
    # 보기 좋게 클러스터 크기 내림차순
    clusters.sort(key=len, reverse=True)
    clusters = tuple(clusters)

    if plot:
        plt.figure(figsize=(8, 4))
        # 레이블은 exp{id}
        leaf_labels = [f"exp{expert_ids[i]}" for i in range(k)]
        # dendrogram 표시 (간단히)
        from scipy.cluster.hierarchy import dendrogram
        dendrogram(Z, labels=leaf_labels)
        ttl = f"Layer {layer_idx} — activated-only CKA (avg-link), k={k}, C={n_use}"
        plt.title(ttl)
        plt.tight_layout()
        plt.show()

    return {
        "clusters": clusters,          # 예: ((0,27,12), (5,11), ...)
        "ids": tuple(expert_ids),      # 군집 입력에 사용된 expert id 목록
        "counts": counts,              # 각 expert의 활성 토큰 수(내림차순 정렬 기준)
        "pair_counts": pair_counts,    # 쌍별 실제 샘플 수
        "order": tuple(order.tolist()),
        "linkage": Z
    }

# ============ 전체 레이어 실행: 최종 반환 포맷 맞춤 ============
def cluster_all_layers_activated(
    expert_outputs, router_logits,
    *,
    n_clusters=8,
    topk_assign=8,
    prob_threshold=None,
    min_activations=1,
    min_samples=8,
    pair_policy='max',
    weight_by_router=False,
    token_stride=1,
    plot=False
):
    """
    모든 레이어에 대해 위의 로직을 적용하고,
    최종 반환을 {"0": ((...), ...), "1": ((...), ...), ...} 형태로 제공.
    """
    L, E, T, D = expert_outputs.shape
    out = {}
    for l in range(L):
        res = cluster_one_layer_activated(
            expert_outputs[l], router_logits[l],
            n_clusters=n_clusters,
            topk_assign=topk_assign, prob_threshold=prob_threshold,
            min_activations=min_activations, min_samples=min_samples,
            pair_policy=pair_policy, weight_by_router=weight_by_router,
            token_stride=token_stride, plot=plot, layer_idx=l
        )
        # 요청한 형태: 레이어 문자열 키 → 클러스터 튜플들의 튜플
        out[str(l)] = res["clusters"]
    return out
