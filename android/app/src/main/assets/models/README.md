# The app's networks

Not in git (~320 MB). Fetch them from the models release:

```bash
scripts/fetch_models.sh
```

| File | What it is |
|---|---|
| `manille_qgru_v1.onnx` | Q-network, trump deals: the value each card gets in the rule agent's blend |
| `manille_qgru_mull_v1.onnx` | Q-network, no-trump (mull) deals |
| `manille_belief_v1.onnx` | belief network, trump deals: which seat holds each unseen card |
| `manille_belief_mull_v1.onnx` | belief network, no-trump deals |
| `manille_fast_policy_v1.bin` | small policy net for the first plies of the play-out (`cpp/fast_policy.h` format) |
| `manille_decl_v1.onnx` | decision layer: trump or mull, by match equity |
| `manille_announce_v1.onnx` | decision layer: knock and knock back |
