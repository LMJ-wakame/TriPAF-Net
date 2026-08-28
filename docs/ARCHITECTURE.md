# TriPAF-Net architecture and evaluated inference path

## Learned network

The active model uses an RGB stream and a prior stream containing dark channel,
bright channel, and a deterministic sky mask. Channel widths for the RGB stream
are `(24, 48, 96, 192, 288)`; prior widths are `(12, 24, 48, 96, 144)`.
`HazeConditionedTriPAF` fuses the streams at all five scales before a
skip-connected decoder.

For luminance `Y = 0.299R + 0.587G + 0.114B`, the scene descriptor is

```text
h = [mean(D), mean(B), mean(M), mean(Y), std(Y)].
```

A learned sigmoid head estimates severity `s`, and the gate context is
`[h, s]`. At scale `l`, the adaptive prior gate is

```text
G_l = 0.5 + 0.5 tanh(a_l) tanh(C_l + S_l + H_l).
```

`a_l` initializes to `0.20`, so this implementation does not claim exact
fixed-gate initialization. The fixed ablation bypasses the learned gate values
and sets both prior and detail gates to `0.5`, while retaining the same modules,
state keys, and parameter count.

The decoder predicts:

- a bounded direct residual image;
- transmission with a lower bound of `0.08`;
- global RGB atmospheric light;
- a spatial blend between direct and physical reconstructions;
- three softmax weights for RGB, learned restoration, and prior reconstruction.

## Output semantics

`TriPAFNetV2.forward(..., return_aux=True)` preserves these primary keys:

- `image`: softmax-fused training output;
- `stable_image`: bounded and model-conditioned inference output;
- `stable_before_color`: output before color/contrast refinement;
- `restoration`, `direct`, `physical`, `prior_reconstruction`;
- `transmission`, `atmospheric_light`, `blend`, `residual`;
- `severity`, `haze_descriptor`, `output_weights`;
- gate and color diagnostic tensors.

`return_aux=False` returns `stable_image`, matching the current model code.

## Evaluated inference

The formal evaluator calls `utils.inference_v2.predict_pil_v2_outputs` and
scores `stable_image`. The model first bounds the learned restoration residual,
protects sky/chroma, restores a fraction of input detail, and applies a
severity-conditioned color/contrast refinement whose coefficients are fixed in
code. Unless `--no-detail-guidance` is supplied, inference then injects
DCP/BCP-derived luminance structure while retaining model chroma.

This makes the formal output a hybrid pipeline:

```text
learned TriPAF network
→ deterministic bounded stability transform
→ learned-severity-conditioned fixed color/contrast operator
→ optional deterministic DCP/BCP luminance-detail guidance
→ stable_image
```

