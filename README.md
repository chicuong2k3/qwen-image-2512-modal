# Qwen-Image-2512 (abliterated text encoder) on Modal

Serverless text-to-image API on [Modal](https://modal.com) running
[Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) with the uncensored text encoder from the
[sci4ai Qwen-Image-2512 (Abliterated Uncensored)](https://huggingface.co/collections/sci4ai/qwen-image-2512-abliterated-uncensored)
collection (`sci4ai/Qwen2.5-VL-7B-Instruct-Ablit`), plus the
[Lightning](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning) distillation LoRA for fast modes.

> The text encoder has its refusal behaviour removed. You are responsible for how the endpoint is used;
> the web endpoint requires Modal proxy auth by default.

## Files

| File | Purpose |
|---|---|
| `app.py` | The deployed app: GPU class, `api` POST endpoint, `modes` GET endpoint, CLI entrypoints |
| `tune.py` | Parameter sweep (16 configs x 4 prompts x 2 seeds) scored with PickScore + HPSv2.1 |
| `realism.py` | Realism experiment: prompt style x text encoder (abliterated vs original) |
| `realism_score.py` | CLIP ViT-H zero-shot "looks like a real photo" probe |
| `bench.py` | Simple latency benchmark against the deployed app |
| `results/` | Text results of the experiments |

## Deploy

```bash
pip install modal
modal token new
modal run app.py::download_weights   # caches ~75 GB of weights in a Modal Volume
modal deploy app.py
```

GPU defaults to H100. Experiments: `QWEN_GPU=A100-80GB`, `QWEN_GPU=L40S QWEN_OFFLOAD=1`, `QWEN_SNAPSHOT=1`.

## Modes

| mode | how | H100 GPU time (1104x1472) | ~cost / image | suggested tokens |
|---|---|---|---|---|
| `turbo` | Lightning 4-step LoRA, 4 steps | ~2.3 s | ~$0.0025 | 1 |
| `fast` (default) | Lightning 4-step LoRA, 6 steps | ~3.2 s | ~$0.0035 | 2 |
| `balanced` | base, 30 steps, cfg 4, 864x1152 | ~11 s | ~$0.012 | 5 |
| `quality` | base, 50 steps, cfg 3 | ~31 s | ~$0.034 | 12 |

Cost uses Modal H100 at $0.001097/s, warm container. A cold start (~75 s load) costs ~$0.08 extra,
and the container idles 60 s (`scaledown_window`) after the last request.

## API

```bash
curl -X POST https://<workspace>--qwen-image-2512-ablit-qwenimage-api.<region>.modal.run \
  -H "Modal-Key: wk-..." -H "Modal-Secret: ws-..." -H "Content-Type: application/json" \
  -d '{"prompt":"old man drinking coffee in a Saigon alley","mode":"quality","aspect":"3:4","rewrite":true,"seed":1}' \
  -o out.png
```

| field | default | notes |
|---|---|---|
| `prompt` | required | |
| `mode` | `fast` | `turbo` / `fast` / `balanced` / `quality` |
| `aspect` | `1:1` | `1:1 16:9 9:16 4:3 3:4 3:2 2:3` |
| `rewrite` | `false` | expand the prompt with the loaded Qwen2.5-VL encoder into a concrete "ordinary photo" description; adds a few seconds; final prompt returned URL-encoded in `X-Prompt` header |
| `enhance` | `true` | append the official suffix `, Ultra HD, 4K, cinematic composition.` |
| `negative_prompt` | `null` | `null` = official Qwen-Image-2512 negative prompt (only used when cfg > 1) |
| `steps`, `cfg`, `seed` | per mode | overrides |

`GET .../modes` returns the mode table as JSON (no GPU).

CLI: `modal run app.py::main --mode quality --rewrite --aspect 3:4 --prompt "..." --out out.png`

## How the defaults were chosen

Sources: the official [model card](https://huggingface.co/Qwen/Qwen-Image-2512) (50 steps, `true_cfg_scale` 4,
the aspect-ratio sizes, the Chinese negative prompt) and the
[Qwen-Image-Lightning reference script](https://github.com/ModelTC/Qwen-Image-Lightning)
(scheduler config with shift = 3, cfg 1, positive suffix). Then measured:

**Sweep** (`tune.py`, `results/tune_summary.txt`): same 4 prompts x 2 seeds for every config, scored by
PickScore and HPSv2.1, z-normalised per prompt/seed.

- Official negative prompt instead of `" "`: largest gain (combined z -0.93 -> -0.28), plus suffix -> +0.41.
- cfg at 50 steps: 3.0 (+0.71) > 2.5 (+0.56) > 4.0 (+0.41) > 5.0 (+0.02). Margins are small.
- Steps with cfg 4: 50 (+0.41) > 40 (+0.26) > 30 (+0.14).
- Lightning: 4-step LoRA at 6 steps scored best (+0.59), above the 8-step LoRA at 10 (+0.39) and 8 (+0.19).
  The suffix helped every Lightning config.

**Realism** (`realism.py`, `results/realism_scores.txt`): the biggest lever for "looks AI" was prompt style.
"85mm, 8k, masterpiece, studio lighting" prompts scored P(real photo) 0.92-0.94; plain concrete descriptions
of an ordinary phone photo scored ~1.00. The abliterated vs original text encoder made no measurable
difference. The CLIP probe saturates, so treat it as a coarse signal and judge the images yourself.

Caveats: 8 images per config; preference models reward aesthetics, not necessarily realism.

## Writing prompts for realistic people

- Describe an ordinary photo: phone camera, overcast or indoor light, unposed moment, imperfect framing.
- Name real imperfections: pores, wrinkles, sun spots, flyaway hair, stubble, dark circles.
- Put concrete background objects in the scene.
- Avoid: `8k`, `masterpiece`, `ultra detailed`, `hyperrealistic`, `studio lighting`, `bokeh`, `85mm f/1.8`.
- Prefer `quality` mode for people; Lightning modes give smoother skin.
- Or send a short prompt with `"rewrite": true`.
