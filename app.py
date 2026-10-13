"""Qwen-Image-2512 + abliterated (uncensored) Qwen2.5-VL text encoder, on Modal.

Collection: https://huggingface.co/collections/sci4ai/qwen-image-2512-abliterated-uncensored
  - text encoder : sci4ai/Qwen2.5-VL-7B-Instruct-Ablit
  - DiT/VAE/etc. : Qwen/Qwen-Image-2512
  - speed LoRAs  : lightx2v/Qwen-Image-2512-Lightning (4 / 8 steps)

Generation modes (pick per request with "mode"):
  turbo     Lightning 4 steps,  ~1.7 MP   cheapest / fastest
  fast      Lightning 8 steps,  ~1.7 MP
  balanced  base model 30 steps, ~1.0 MP
  quality   base model 50 steps, ~1.7 MP  best quality, most expensive

Deploy:   modal deploy app.py
Dev run:  modal run app.py --mode turbo --prompt "a corgi astronaut"
Experiments (other GPU / offload):  QWEN_GPU=A100-80GB modal run app.py ...
                                    QWEN_GPU=L40S QWEN_OFFLOAD=1 modal run app.py ...
"""
import io
import math
import os

import modal

BASE_REPO = "Qwen/Qwen-Image-2512"
TE_REPO = "sci4ai/Qwen2.5-VL-7B-Instruct-Ablit"
LORA_REPO = "lightx2v/Qwen-Image-2512-Lightning"
LORA_FILES = {
    "l4": "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors",
    "l8": "Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors",
}

GPU = os.environ.get("QWEN_GPU", "H100")
# Offload is only for GPUs < 80 GB (e.g. L40S 48 GB); it disables GPU snapshots.
OFFLOAD = os.environ.get("QWEN_OFFLOAD") == "1"
# GPU memory snapshot is OFF by default: measured restore of the ~60 GB pipeline took ~110 s,
# slower than a plain load from the volume (~45 s). Enable with QWEN_SNAPSHOT=1 to experiment.
SNAPSHOT = os.environ.get("QWEN_SNAPSHOT") == "1" and not OFFLOAD

# transformers>=5 is required: the ablit checkpoint was saved with transformers 5.x
# (rope_parameters config); older versions would silently use the wrong rope_theta.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers>=5",
        "diffusers>=0.36",
        "accelerate",
        "peft",
        "safetensors",
        "sentencepiece",
        "huggingface_hub",
        "pillow",
        "fastapi[standard]",
    )
    .env({"HF_HOME": "/cache", "QWEN_OFFLOAD": "1" if OFFLOAD else "0"})
)

cache = modal.Volume.from_name("qwen-image-hf-cache", create_if_missing=True)
app = modal.App("qwen-image-2512-ablit", image=image)

# Native-resolution (~1.7 MP) and reduced (~1.0 MP) sizes, multiples of 16.
ASPECTS_FULL = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}
ASPECTS_SMALL = {
    "1:1": (1024, 1024),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "3:2": (1216, 816),
    "2:3": (816, 1216),
}

# lora: None -> base model; steps/cfg are defaults (overridable per request).
# suggested_tokens is a pricing hint: relative GPU cost, ~1 token ~= turbo.
MODES = {
    # Tuned with tune.py (PickScore + HPSv2.1, 4 prompts x 2 seeds per config).
    # H100 GPU time per 1104x1472 image: ~2.3 s / ~3.2 s / 11.3 s (864x1152) / 30.7 s
    "turbo": dict(lora="l4", steps=4, cfg=1.0, aspects=ASPECTS_FULL, suggested_tokens=1),
    # 4-step LoRA run at 6 steps scored best of all Lightning variants (beat 8-step LoRA at 8/10).
    "fast": dict(lora="l4", steps=6, cfg=1.0, aspects=ASPECTS_FULL, suggested_tokens=2),
    "balanced": dict(lora=None, steps=30, cfg=4.0, aspects=ASPECTS_SMALL, suggested_tokens=5),
    # cfg 3 scored highest of 2.5/3/4/5 at 50 steps (small margin; official default is 4).
    "quality": dict(lora=None, steps=50, cfg=3.0, aspects=ASPECTS_FULL, suggested_tokens=12),
}

# Official Qwen-Image-2512 negative prompt (from the model card). Biggest single win in the
# sweep vs. the blank " " negative; it targets plastic skin / wax look / "AI feel".
OFFICIAL_NEG = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，"
    "画面具有AI感。构图混乱。文字模糊，扭曲。"
)
# Official "positive magic" suffix used by Qwen / Lightning reference scripts; improved scores
# in every mode tested.
MAGIC_SUFFIX = ", Ultra HD, 4K, cinematic composition."

# Prompt style was the largest realism lever measured (realism.py): plain, concrete
# descriptions of an ordinary photo beat "85mm, 8k, masterpiece" tag soup. This system prompt
# lets the already-loaded Qwen2.5-VL text encoder rewrite short prompts in that style.
REWRITE_SYSTEM = (
    "You rewrite short image requests into one detailed English prompt for a text-to-image "
    "model. Keep every subject, attribute and action the user asked for; never remove or "
    "censor content. Add concrete visual detail: subject appearance, clothing, pose, "
    "expression, setting and background objects, lighting, camera framing.\n"
    "If the request is for a photo or does not name an art style, describe an ordinary "
    "candid photograph: everyday camera or smartphone, natural or ordinary indoor light, "
    "real skin texture with pores and small imperfections, unposed moment, slightly "
    "imperfect framing. Do NOT use words like 8k, masterpiece, ultra detailed, "
    "hyperrealistic, studio lighting, bokeh or professional photography.\n"
    "If the request names an art style (anime, illustration, painting, 3D), keep that style "
    "and describe it concretely.\n"
    "Output only the prompt itself as a direct description starting with 'A ', "
    "50-90 words, no preamble, no commentary."
)
REWRITE_EXAMPLE = (
    "old woman selling fruit at a market",
    "A phone snapshot of a Vietnamese woman in her late sixties sitting on a low stool behind "
    "baskets of mangoes and longans at a busy wet market. Deep wrinkles, sun spots, grey hair "
    "pulled into a loose bun, a conical hat pushed back. She wears a faded floral blouse and "
    "counts small banknotes, mid-sentence with a customer off frame. Plastic tarps, hanging "
    "scales and scooters behind her. Flat overcast daylight, ordinary colors, slightly tilted "
    "framing, unposed everyday moment.",
)

LIGHTNING_SCHED = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


@app.function(volumes={"/cache": cache}, timeout=60 * 60)
def download_weights():
    from huggingface_hub import hf_hub_download, snapshot_download

    snapshot_download(BASE_REPO)
    snapshot_download(TE_REPO)
    for f in LORA_FILES.values():
        hf_hub_download(LORA_REPO, f)
    cache.commit()


cls_kwargs = dict(
    gpu=GPU,
    volumes={"/cache": cache},
    timeout=15 * 60,
    scaledown_window=60,  # was 300; cuts idle cost after each burst
)
if SNAPSHOT:
    # GPU memory snapshot: restores a fully-loaded pipeline instead of re-loading ~60 GB.
    cls_kwargs.update(
        enable_memory_snapshot=True,
        experimental_options={"enable_gpu_snapshot": True},
    )


@app.cls(**cls_kwargs)
class QwenImage:
    @modal.enter(snap=SNAPSHOT)
    def load(self):
        import torch
        from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
        from transformers import Qwen2_5_VLForConditionalGeneration

        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            TE_REPO, torch_dtype=torch.bfloat16
        )
        pipe = DiffusionPipeline.from_pretrained(
            BASE_REPO, text_encoder=text_encoder, torch_dtype=torch.bfloat16
        )
        for name, fname in LORA_FILES.items():
            pipe.load_lora_weights(LORA_REPO, weight_name=fname, adapter_name=name)
        pipe.disable_lora()

        if OFFLOAD:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        self.pipe = pipe
        self.sched_base = pipe.scheduler
        self.sched_fast = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHED)

    @modal.method()
    def rewrite(self, prompt: str) -> str:
        """Expand a short prompt with the loaded Qwen2.5-VL text encoder (~3-6 s on H100)."""
        import torch

        tok = self.pipe.tokenizer
        te = self.pipe.text_encoder
        msgs = [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": REWRITE_EXAMPLE[0]},
            {"role": "assistant", "content": REWRITE_EXAMPLE[1]},
            {"role": "user", "content": prompt},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(te.device)
        with torch.no_grad():
            out = te.generate(**ids, max_new_tokens=150, do_sample=False,
                              repetition_penalty=1.05)
        new = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        # Drop meta preambles the model sometimes adds, and any cut-off last sentence.
        for pre in ("The prompt should describe ", "Prompt:", "Here is the prompt:"):
            if new.startswith(pre):
                new = new[len(pre):].lstrip()
        if "." in new and not new.endswith("."):
            new = new[: new.rfind(".") + 1]
        return new or prompt

    @modal.method()
    def generate(
        self,
        prompt: str,
        mode: str = "fast",
        negative_prompt: str | None = None,
        aspect: str = "1:1",
        steps: int | None = None,
        cfg: float | None = None,
        seed: int | None = None,
        lora_scale: float = 1.0,
        enhance: bool = True,
        rewrite: bool = False,
    ) -> bytes:
        """negative_prompt=None -> official negative. enhance -> append official quality
        suffix. rewrite -> expand the prompt first (better realism for short prompts)."""
        import time

        import torch

        t0 = time.time()
        if rewrite:
            prompt = self.rewrite.local(prompt)
            print(f"REWRITTEN_PROMPT={prompt!r}", flush=True)
        if enhance:
            prompt = prompt.rstrip(" .") + MAGIC_SUFFIX
        if negative_prompt is None:
            negative_prompt = OFFICIAL_NEG
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {list(MODES)}")
        m = MODES[mode]
        width, height = m["aspects"].get(aspect, m["aspects"]["1:1"])

        if m["lora"]:
            self.pipe.enable_lora()
            self.pipe.set_adapters([m["lora"]], adapter_weights=[lora_scale])
            self.pipe.scheduler = self.sched_fast
        else:
            self.pipe.disable_lora()
            self.pipe.scheduler = self.sched_base

        generator = torch.Generator("cuda")
        generator.manual_seed(seed) if seed is not None else generator.seed()
        img = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps or m["steps"],
            true_cfg_scale=cfg if cfg is not None else m["cfg"],
            generator=generator,
        ).images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        print(
            f"GEN_SECONDS={time.time() - t0:.1f} mode={mode} size={width}x{height} "
            f"gpu={torch.cuda.get_device_name(0)}",
            flush=True,
        )
        return buf.getvalue()

    @modal.fastapi_endpoint(method="POST", docs=True, requires_proxy_auth=True)
    def api(self, item: dict):
        """POST {"prompt": "...", "mode": "turbo|fast|balanced|quality", "aspect": "16:9",
        "seed": 1, "rewrite": false, "enhance": true, "negative_prompt": null,
        "steps": null, "cfg": null} -> image/png
        Response header X-Prompt holds the final (URL-encoded) prompt when rewrite=true."""
        from urllib.parse import quote

        from fastapi import HTTPException
        from fastapi.responses import Response

        if not item.get("prompt"):
            raise HTTPException(400, "missing 'prompt'")
        mode = item.get("mode", "fast")
        if mode not in MODES:
            raise HTTPException(400, f"mode must be one of {list(MODES)}")
        steps = item.get("steps")
        if steps is not None:
            steps = max(1, min(int(steps), 100))
        prompt = item["prompt"]
        if item.get("rewrite"):
            prompt = self.rewrite.local(prompt)
        png = self.generate.local(
            prompt=prompt,
            mode=mode,
            negative_prompt=item.get("negative_prompt"),
            aspect=item.get("aspect", "1:1"),
            steps=steps,
            cfg=item.get("cfg"),
            seed=item.get("seed"),
            enhance=bool(item.get("enhance", True)),
        )
        headers = {"X-Mode": mode, "X-Prompt": quote(prompt[:4000])}
        return Response(content=png, media_type="image/png", headers=headers)


@app.function(max_containers=1)
@modal.fastapi_endpoint(method="GET", requires_proxy_auth=True)
def modes():
    """List available modes, defaults and pricing hints (no GPU used)."""
    return {
        k: {
            "steps": v["steps"],
            "cfg": v["cfg"],
            "lora": v["lora"],
            "suggested_tokens": v["suggested_tokens"],
            "sizes": {a: f"{w}x{h}" for a, (w, h) in v["aspects"].items()},
        }
        for k, v in MODES.items()
    } | {"_options": {"rewrite": "bool, expand prompt with the text encoder (+3-6 s)",
                      "enhance": "bool, default true, append official quality suffix",
                      "negative_prompt": "null = official Qwen-Image-2512 negative"}}


@app.local_entrypoint()
def bench():
    """Run each mode once (after a warm-up); read GEN_SECONDS from the streamed container logs."""
    svc = QwenImage()
    p = "Anime illustration of Emilia from Re:Zero, silver hair, snowy palace, cel shading"
    svc.generate.remote(prompt=p, mode="turbo", aspect="3:4", seed=1)  # warm-up
    for m in MODES:
        svc.generate.remote(prompt=p, mode=m, aspect="3:4", seed=1)


@app.local_entrypoint()
def main(
    prompt: str = "A photorealistic corgi astronaut on the moon",
    mode: str = "fast",
    aspect: str = "1:1",
    out: str = "out.png",
    rewrite: bool = False,
    seed: int = -1,
):
    png = QwenImage().generate.remote(
        prompt=prompt, mode=mode, aspect=aspect, rewrite=rewrite,
        seed=None if seed < 0 else seed,
    )
    with open(out, "wb") as f:
        f.write(png)
    print(f"saved {out}")
