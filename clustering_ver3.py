import numpy as np
from scipy.special import softmax
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

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
                                   min_activations=1, cap=None):
    """
    해당 레이어에서 '활성 토큰 수 >= min_activations'인 expert를 활성도(토큰 수) 내림차순으로 정렬.
    cap가 주어지면 상위 cap명만 반환 -> 여기서 cap=8로 쓰면 '활성된 8명만'이 입력으로 들어감.
    """
    act = get_active_token_mask_layer(router_logits_layer, topk_assign, prob_threshold)  # (E,T)
    counts_all = act.sum(axis=1)  # (E,)
    ids = np.where(counts_all >= min_activations)[0]
    counts = counts_all[ids]
    order = np.argsort(-counts)            # 활성 토큰 수 많은 순
    ids = ids[order].tolist()
    counts = counts[order].astype(int)
    if cap is not None and len(ids) > cap:
        ids = ids[:cap]
        counts = counts[:cap]
    return ids, counts

# 클러스터 내부 평균 거리(옵션: 쌍별 가중치로 평균)
def cluster_intra_average_distance_from_condensed(
    condensed,              # 길이 k*(k-1)/2 의 1-CKA 거리 벡터
    members_idx,            # 이 클러스터의 멤버 "행렬 인덱스" 리스트 (0..k-1)
    *,
    weights_condensed=None, # (선택) pair_counts 같은 쌍별 가중치(같은 길이)
    use_weights=False,
    singleton_as_nan=True,  # 멤버 1명일 때 NaN(기본) or 0.0
    eps=1e-12
):
    """
    클러스터 C의 내부 평균거리:
      d̄(C) = (1 / |C|(|C|-1)/2) * sum_{a<b, a,b∈C} d(a,b)
    use_weights=True면 쌍별 가중 평균(예: 교집합 토큰 수로 가중)
    """
    idx = np.asarray(members_idx, dtype=int)
    if idx.size <= 1:
        return np.nan if singleton_as_nan else 0.0

    D = squareform(condensed)  # (k,k)
    sub = D[np.ix_(idx, idx)]
    # 상삼각(대각 제외)만 취함
    iu = np.triu_indices(sub.shape[0], k=1)
    vals = sub[iu]

    if not use_weights:
        return float(np.nanmean(vals))

    assert weights_condensed is not None, "use_weights=True면 weights_condensed 필요"
    W = squareform(weights_condensed)
    wsub = W[np.ix_(idx, idx)]
    wvals = wsub[iu]
    num = float((vals * wvals).sum())
    den = float(wvals.sum()) + eps
    return num / den

# ============ CKA (linear) ============
def compute_linear_cka(X, Y, sample_weights=None, eps=1e-12):
    """
    X: (N, Dx), Y: (N, Dy) — 샘플축: 토큰들(동시 활성 교집합)
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
    min_samples=8,          # 두 expert의 '동시 활성' 교집합 최소 길이
    pair_policy='max',      # 'max' -> 교집합 < min이면 거리=1.0, 'nan' -> np.nan
    weight_by_router=False,
    token_stride=1
):
    """
    반환: condensed distances (1-CKA), pair_counts  (길이 k*(k-1)/2)
    거리 계산은 '두 expert가 동시에 활성'인 토큰 교집합에서만 수행.
    """
    E, T, D = expert_outputs_layer.shape
    tok_idx = np.arange(T)[::max(1, token_stride)]
    act = get_active_token_mask_layer(router_logits_layer, topk_assign, prob_threshold)  # (E,T)
    act = act[:, tok_idx]  # (E, Ts)

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

# ============ 레이어별 클러스터링: '활성 상위 8명만' 입력 ============
def cluster_one_layer_activated(
    expert_outputs_layer,   # (E, T, D)
    router_logits_layer,    # (T, E)
    *,
    n_experts=8,            # ★ 활성 상위 n명만 사용 (기본 8)
    n_clusters=3,           # 결과 클러스터 개수 컷
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
    이 레이어에서 '활성된 expert' 중 상위 n_experts명만 뽑아
    동시 활성 교집합에서 CKA로 거리 계산 → average linkage → n_clusters로 컷.
    """
    expert_ids, counts = collect_activated_ids_in_layer(
        router_logits_layer, topk_assign, prob_threshold,
        min_activations=min_activations, cap=n_experts  # ← 여기서 8명으로 제한
    )
    k = len(expert_ids)
    if k < 2:
        return {"clusters": tuple(), "ids": tuple(), "counts": np.asarray(counts), "order": tuple(), "linkage": None}

    condensed, pair_counts = pairwise_cka_condensed_layer_activated(
        expert_outputs_layer, router_logits_layer, expert_ids,
        topk_assign=topk_assign, prob_threshold=prob_threshold,
        min_samples=min_samples, pair_policy=pair_policy,
        weight_by_router=weight_by_router, token_stride=token_stride
    )
    if np.isnan(condensed).any():
        raise ValueError("NaN distance 발생: pair_policy='max' 사용 또는 활성 기준을 완화하세요.")

    Z = linkage(condensed, method='average')
    order = leaves_list(Z)

    n_use = min(n_clusters, k)
    labels = fcluster(Z, t=n_use, criterion='maxclust')  # 1..n_use

    clusters = []
    for c in range(1, n_use + 1):
        members = [expert_ids[i] for i in range(k) if labels[i] == c]
        clusters.append(tuple(members))
    clusters.sort(key=len, reverse=True)
    clusters = tuple(clusters)

    if plot:
        from scipy.cluster.hierarchy import dendrogram
        plt.figure(figsize=(8, 4))
        leaf_labels = [f"exp{expert_ids[i]}" for i in range(k)]
        dendrogram(Z, labels=leaf_labels)
        ttl = f"Layer {layer_idx} — activated-only CKA (avg-link), k={k}, C={n_use}"
        plt.title(ttl)
        plt.tight_layout()
        plt.show()
        # expert_ids의 포지션(0..k-1)으로 바꾸기 위한 매핑
    id_to_pos = {eid: i for i, eid in enumerate(expert_ids)}
    # condensed/pair_counts는 위에서 만든 쌍별 1-CKA와 샘플 수
    intra_avg_unweighted = []
    #intra_avg_weighted   = []

    for cl in clusters:
        # cl은 실제 expert id들의 튜플 → 0..k-1 인덱스로 변환
        memb_pos = [id_to_pos[eid] for eid in cl]
        d_unw = cluster_intra_average_distance_from_condensed(
            condensed, memb_pos, use_weights=False
        )
        #d_w = cluster_intra_average_distance_from_condensed(
        #    condensed, memb_pos,
        #    weights_condensed=pair_counts, use_weights=True
        #)
        intra_avg_unweighted.append(d_unw)
        #intra_avg_weighted.append(d_w)

    return {
        "clusters": clusters,                # 예: ((0,27,12), (5,11), ...)
        "ids": tuple(expert_ids),
        "counts": counts,
        "pair_counts": pair_counts,          # 쌍별 동시활성 샘플 수
        "condensed": condensed,              # 쌍별 1-CKA
        "intra_avg_unweighted": tuple(intra_avg_unweighted),  # 각 클러스터의 내부 평균거리
        #"intra_avg_weighted":   tuple(intra_avg_weighted),    # (교집합 샘플 수 가중)
        "order": tuple(order.tolist()),
        "linkage": Z
    }

# ============ 전체 레이어 실행: {"0": ((...), ...), ...} ============
def cluster_all_layers_activated(
    expert_outputs, router_logits,
    *,
    n_experts=8,            # ★ 레이어마다 활성 상위 8명만
    n_clusters=3,
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
    모든 레이어에 대해 활성 상위 n_experts명만 사용하여 군집.
    최종 반환: {"0": ((...), ...), "1": ((...), ...), ...}
    """
    L, E, T, D = expert_outputs.shape
    out = {}
    for l in range(L):
        res = cluster_one_layer_activated(
            expert_outputs[l], router_logits[l],
            n_experts=n_experts, n_clusters=n_clusters,
            topk_assign=topk_assign, prob_threshold=prob_threshold,
            min_activations=min_activations, min_samples=min_samples,
            pair_policy=pair_policy, weight_by_router=weight_by_router,
            token_stride=token_stride, plot=plot, layer_idx=l
        )
        out[str(l)] = res["clusters"]
        pairs = list(zip(res["clusters"], res["intra_avg_unweighted"]))
    return out
