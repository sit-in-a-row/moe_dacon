### 실행 예시(ver1)
```python
simple = cluster_all_layers_simple_output(
    expert_outputs, router_logits,
    n_experts=128, topk_assign=128, min_activations=0,
    threshold=0.8,
    distance_mode="centroid",
    pooling_weight="router",
    verbose=True
)

simple = cluster_all_layers_simple_output(
    expert_outputs, router_logits,
    n_experts=128, topk_assign=128, min_activations=0,
    threshold=0.8,
    distance_mode="pairwise",   #pairwise or cka
    pooling_weight="none",
    verbose=True
)

```

### 실행 예시(ver2)
```python
simple = cluster_all_layers_simple_output(
    expert_outputs, router_logits,
    n_experts=128, topk_assign=8, min_activations=1,
    threshold=0.8,
    distance_mode="centroid",   # 중심 간 L2
    pooling_weight="router",    # 가중 중심
    verbose=False               # 내용보고 싶으면 True ㄱㄱ
)

simple = cluster_all_layers_simple_output(
    expert_outputs, router_logits,
    n_experts=128, topk_assign=8, min_activations=1,
    threshold=0.8,
    distance_mode="pairwise",   # 모든 쌍 L2 평균
    pooling_weight="none",      # router score는 풀링에 쓰지 않음
    verbose=True                # 내용보고 싶으면 True ㄱㄱ 
)
```

```python
### Output 형태
print(out)
# {'0': ((3, 27, 12), (5, 11), (7,)), '1': ((2, 9), (0, 4, 6), (1, 3)), ...}
```
