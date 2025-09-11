### 실행 예시

```python
out = cluster_all_layers_simple_output(
    expert_outputs, router_logits,
    n_experts=8,
    threshold=0.8,          # L2 기준
    distance_mode="pairwise" # 또는 "centroid"
)
```
```python
### Output 형태
print(out)
# {'0': ((3, 27, 12), (5, 11), (7,)), '1': ((2, 9), (0, 4, 6), (1, 3)), ...}
```
