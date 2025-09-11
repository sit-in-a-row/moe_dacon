import numpy as np
from scipy.special import softmax
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import matplotlib.pyplot as plt

# ============ 공통 유틸 ============
def _center_rows(X):
    return X - X.mean(axis=0, keepdims=True)

# 레이어별 활성 마스크 (토큰 기준 top-k OR 확률 임계)
def get_active_token_mask_layer(router_logits_layer, topk_assign=8, prob_threshold=None):
    """
    router_logits_layer: (T, E)
    반환: (E, T) bool
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

# 입력에서 실제 "활성된" (layer, expert)만 수집
def collect_activated_ids_across_layers(router_logits, topk_assign=8, prob_threshold=None,
                                        min_activations=1, cap=None, sort_desc=True):
    """
    router_logits: (L, T, E)
    반환:
      ids: [(l,e)] — 입력에서 활성(≥min_activations)된 expert들
      counts: 각 (l,e)의 활성 토큰 개수
    """
    L, T, E = router_logits.shape
    ids, counts = [], []
    for l in range(L):
        act = get_active_token_mask_layer(router_logits[l], topk_assign, prob_threshold)  # (E,T)
        c = act.sum(axis=1)  # (E,)
        for e in range(E):
            if c[e] >= min_activations:
                ids.append((l, e))
                counts.append(int(c[e]))
    counts = np.asarray(counts, int)
    if sort_desc and len(ids) > 0:
        order = np.argsort(-counts)
        ids = [ids[i] for i in order]
        counts = counts[order]
    if (cap is not None) and (len(ids) > cap):
        ids = ids[:cap]
        counts = counts[:cap]
    return ids, counts

# ============ CKA (linear) ============
def compute_linear_cka(X, Y, sample_weights=None, eps=1e-12):
    """
    X: (N, Dx), Y: (N, Dy) — 샘플축: 토큰
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

# ============ 전역 쌍별 CKA (활성 토큰 교집합만) ============
def pairwise_cka_condensed_global_activated(
    expert_outputs,         # (L, E, T, D)
    router_logits,          # (L, T, E)
    ids,                    # list[(l,e)] 길이 k — 활성 expert만
    *,
    topk_assign=8,
    prob_threshold=None,
    min_samples=8,
    pair_policy='max',      # 'max' -> 교집합<min이면 거리=1.0
    weight_by_router=False, # 활성-only 
    token_stride=1
):
    """
    반환: (condensed distances (1-CKA), pair_counts)  where len = k*(k-1)/2
    - 거리 계산은 오직 '동시 활성 토큰 교집합'에서만.
    - 교집합이 min_samples 미만이면:
        'max'  : 거리=1.0 (비활성 토큰은 절대 사용하지 않음)
        'nan'  : np.nan (linkage가 깨질 수 있음)
    """
    L, E, T, D = expert_outputs.shape
    tok_idx = np.arange(T)[::max(1, token_stride)]
    # 레이어별 활성 마스크 캐시
    act_cache = {}
    probs_cache = {} if weight_by_router else None

    mats, acts, wts = [], [], []
    for (l, e) in ids:
        if l not in act_cache:
            act_cache[l] = get_active_token_mask_layer(router_logits[l], topk_assign, prob_threshold)  # (E,T)
        mask = act_cache[l][e][tok_idx]  # (Ts,)
        acts.append(mask)
        mats.append(expert_outputs[l, e, tok_idx, :])  # (Ts, D)
        if weight_by_router:
            if l not in probs_cache:
                probs_cache[l] = softmax(router_logits[l], axis=-1)  # (T,E)
            wts.append(probs_cache[l][tok_idx, e])  # (Ts,)
        else:
            wts.append(None)

    k = len(ids)
    dists, counts = [], []
    for i in range(k-1):
        for j in range(i+1, k):
            mask = acts[i] & acts[j]
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
                    dists.append(1.0); counts.append(idx.size)  # 활성-only 유지, 보수적 거리
                else:  # 'nan'
                    dists.append(np.nan); counts.append(idx.size)
    return np.asarray(dists, float), np.asarray(counts, int)

# ============ HC-SMoE: average linkage ============
def hierarchical_cluster_HC(condensed, labels=None, plot=False, title=None):
    Z = linkage(condensed, method='average')
    order = leaves_list(Z)
    if plot:
        plt.figure(figsize=(10, 5))
        dendrogram(Z, labels=None if labels is None else [labels[i] for i in range(len(labels))])
        plt.title(title or "Fubao")
        plt.tight_layout()
        plt.show()
    return Z, order

# ============ 상위 API: 전역(레이어 통합) activated-only CKA ============
def cluster_experts_global_activated(
    expert_outputs, router_logits,
    *,
    plot=False,
    cap=None,                    # None이면 활성된 전부 사용
    topk_assign=8,
    prob_threshold=None,
    min_activations=1,           # (l,e) 포함 최소 활성 토큰 수
    min_samples=8,               # CKA용 두 expert 교집합 최소 샘플 수
    pair_policy='max',           # 'max' or 'nan'
    weight_by_router=False, 
    token_stride=1
):
    """
    입력에서 실제 활성된 (layer, expert)만 모아, 동시 활성 토큰 교집합에서만 CKA를 계산.
    비활성 토큰 표현은 전혀 사용하지 않음.
    """
    L, E, T, D = expert_outputs.shape
    ids, counts = collect_activated_ids_across_layers(
        router_logits, topk_assign, prob_threshold,
        min_activations=min_activations, cap=cap, sort_desc=True
    )
    if len(ids) < 7:
        return {"message": "활성된 expert가 7개 미만입니다.", "ids": ids, "counts": counts}

    labels = [f"L{l}:exp{int(e)}" for (l, e) in ids]

    condensed, pair_counts = pairwise_cka_condensed_global_activated(
        expert_outputs, router_logits, ids,
        topk_assign=topk_assign, prob_threshold=prob_threshold,
        min_samples=min_samples, pair_policy=pair_policy,
        weight_by_router=weight_by_router, token_stride=token_stride
    )

    if np.isnan(condensed).any():
        raise ValueError("condensed distance에 NaN이 포함->pair_policy='max'를 사용하거나, min_samples/활성 기준을 조정하슈.")

    Z, order = hierarchical_cluster_HC(
        condensed, labels=labels, plot=plot,
        title=f"Global clustering. n={len(ids)}"
    )

    return {
        "metric": "cka(activated-only)",
        "n_selected": len(ids),
        "ids": ids,                   # [(layer, expert)]
        "activation_counts": counts,  # 각 expert의 활성 토큰 수
        "pair_counts": pair_counts,   # 쌍별 실제 사용 샘플 수
        "labels": labels,
        "linkage": Z,
        "order": order
    }
