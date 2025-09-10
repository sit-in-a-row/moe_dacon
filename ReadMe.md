### 실행 예시

```python
clusters_by_layer = cluster_all_layers_activated(
    expert_outputs, router_logits,
    n_experts=8,        # 각 레이어에서 활성 상위 8명만
    n_clusters=3,       # 결과 클러스터 개수
    topk_assign=8,
    min_activations=1,
    min_samples=8,
    pair_policy='max',
    weight_by_router=False,
    plot=False
)

# 예시 출력
for layer_key, clusters in clusters_by_layer.items():
    print(f"[Layer {layer_key}] clusters: {clusters}")
```
```python
### Output 형태
{
  "0": ((3, 27, 12), (5, 11), (7,)),
  "1": ((2, 9), (0, 4, 6), (1, 3)),
  ...
}
```
