"""Parameter sweep for the deployed qwen-image-2512-ablit app, scored automatically.

  modal run --detach tune.py   -> runs on Modal, writes to Volume "qwen-tune-out":
                                  images, scores.csv, summary.txt, sheet_*.jpg
  modal volume get qwen-tune-out / tune_out   -> download results

Every config renders the SAME prompts with the SAME seeds, so differences come from the
parameters only. Each image is scored by two learned human-preference models:
  - PickScore  (yuvalkirstain/PickScore_v1)
  - HPS v2.1   (xswu/HPSv2)
Scores are computed against the plain prompt (without any quality suffix).
"""
import csv
import io
import json
import os

import modal

OFFICIAL_NEG = (
    "ä½Žåˆ†è¾¨çŽ‡ï¼Œä½Žç”»è´¨ï¼Œè‚¢ä½“ç•¸å½¢ï¼Œæ‰‹æŒ‡ç•¸å½¢ï¼Œç”»é¢è¿‡é¥±å’Œï¼Œèœ¡åƒæ„Ÿï¼Œäººè„¸æ— ç»†èŠ‚ï¼Œè¿‡åº¦å…‰æ»‘ï¼Œ"
    "ç”»é¢å…·æœ‰AIæ„Ÿã€‚æž„å›¾æ··ä¹±ã€‚æ–‡å­—æ¨¡ç³Šï¼Œæ‰­æ›²ã€‚"
)
MAGIC = ", Ultra HD, 4K, cinematic composition."

PROMPTS = {
    "portrait": "Photorealistic portrait photo of a young Vietnamese woman in her mid-20s, natural makeup, long black hair, soft smile, beige linen shirt, standing on a Hanoi street at golden hour, shallow depth of field, 85mm lens, natural skin texture",
    "elder": "Candid street photograph of an elderly Vietnamese man with a weathered, wrinkled face and white stubble drinking iced coffee at a small plastic table in a Saigon alley, morning light, 35mm documentary style",
    "anime": "Anime key visual of Emilia from Re:Zero, long silver hair with a white flower ornament, violet eyes, white and purple dress, snowy fantasy palace courtyard, falling snowflakes, soft magical glow, clean lineart, cel shading",
    "dog": "An ultra-realistic close-up of a golden retriever outdoors under soft daylight, individual fur strands visible, moist expressive eyes, softly blurred green park background",
}
SEEDS = [11, 22]
MAX_INFLIGHT = 6

# name: (mode, steps, cfg, negative, magic, lora_scale)
CONFIGS = {
    # --- base model (quality family) ---
    "Q50_cfg4_neg-blank": ("quality", 50, 4.0, " ", False, 1.0),          # what was deployed
    "Q50_cfg4_neg-off": ("quality", 50, 4.0, OFFICIAL_NEG, False, 1.0),
    "Q50_cfg4_neg-off_magic": ("quality", 50, 4.0, OFFICIAL_NEG, True, 1.0),  # official recipe
    "Q50_cfg2.5_off_magic": ("quality", 50, 2.5, OFFICIAL_NEG, True, 1.0),
    "Q50_cfg3_off_magic": ("quality", 50, 3.0, OFFICIAL_NEG, True, 1.0),
    "Q50_cfg5_off_magic": ("quality", 50, 5.0, OFFICIAL_NEG, True, 1.0),
    "Q40_cfg4_off_magic": ("quality", 40, 4.0, OFFICIAL_NEG, True, 1.0),
    "Q30_cfg4_off_magic": ("quality", 30, 4.0, OFFICIAL_NEG, True, 1.0),
    # --- Lightning (cfg 1 => negative prompt unused) ---
    "L4x4": ("turbo", 4, 1.0, " ", False, 1.0),                          # deployed turbo
    "L4x4_magic": ("turbo", 4, 1.0, " ", True, 1.0),
    "L4x6_magic": ("turbo", 6, 1.0, " ", True, 1.0),
    "L4x4_s0.85_magic": ("turbo", 4, 1.0, " ", True, 0.85),
    "L8x8": ("fast", 8, 1.0, " ", False, 1.0),                           # deployed fast
    "L8x8_magic": ("fast", 8, 1.0, " ", True, 1.0),
    "L8x6_magic": ("fast", 6, 1.0, " ", True, 1.0),
    "L8x10_magic": ("fast", 10, 1.0, " ", True, 1.0),
}

scorer_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", "transformers==4.46.3", "open_clip_torch", "huggingface_hub", "pillow"
).env({"HF_HOME": "/cache"})
cache = modal.Volume.from_name("qwen-image-hf-cache", create_if_missing=True)
app = modal.App("qwen-image-tune", image=scorer_image)


@app.cls(gpu="A10G", volumes={"/cache": cache}, timeout=30 * 60)
class Scorer:
    @modal.enter()
    def load(self):
        import open_clip
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoProcessor

        self.pk_proc = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.pk = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().half().cuda()

        model, _, pre = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
        ck = torch.load(hf_hub_download("xswu/HPSv2", "HPS_v2.1_compressed.pt"), map_location="cpu")
        model.load_state_dict(ck["state_dict"])
        self.hps, self.hps_pre = model.eval().half().cuda(), pre
        self.hps_tok = open_clip.get_tokenizer("ViT-H-14")
        cache.commit()

    @modal.method()
    def score(self, png: bytes, prompt: str) -> dict:
        import torch
        from PIL import Image

        img = Image.open(io.BytesIO(png)).convert("RGB")
        with torch.no_grad():
            ii = self.pk_proc(images=img, return_tensors="pt").to("cuda")
            ti = self.pk_proc(text=prompt, padding=True, truncation=True, max_length=77,
                              return_tensors="pt").to("cuda")
            ie = self.pk.get_image_features(**{k: v.half() for k, v in ii.items()})
            te = self.pk.get_text_features(**ti)
            ie, te = ie / ie.norm(dim=-1, keepdim=True), te / te.norm(dim=-1, keepdim=True)
            pick = (self.pk.logit_scale.exp() * (ie * te).sum()).item()

            x = self.hps_pre(img).unsqueeze(0).half().cuda()
            t = self.hps_tok([prompt]).cuda()
            f_i = self.hps.encode_image(x, normalize=True)
            f_t = self.hps.encode_text(t, normalize=True)
            hps = (f_i * f_t).sum().item()
        return {"pick": pick, "hps": hps}

    @modal.method()
    def sheet(self, images: list[bytes], labels: list[str], cols: int) -> bytes:
        from PIL import Image, ImageDraw

        tw, th = 368, 491  # thumbnails of 1104x1472
        rows = (len(images) + cols - 1) // cols
        out = Image.new("RGB", (cols * tw, rows * (th + 22)), "white")
        d = ImageDraw.Draw(out)
        for i, (b, lab) in enumerate(zip(images, labels)):
            im = Image.open(io.BytesIO(b)).convert("RGB").resize((tw, th))
            x, y = (i % cols) * tw, (i // cols) * (th + 22)
            out.paste(im, (x, y + 22))
            d.text((x + 4, y + 4), lab, fill="black")
        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=88)
        return buf.getvalue()


outvol = modal.Volume.from_name("qwen-tune-out", create_if_missing=True)
OUT = "/out"


@app.function(volumes={OUT: outvol}, timeout=3 * 3600)
def sweep() -> str:
    """Runs entirely on Modal (survives local disconnects). Resumable: existing PNGs are reused."""
    gen = modal.Cls.from_name("qwen-image-2512-ablit", "QwenImage")()
    scorer = Scorer()

    tasks = [
        (cname, pname, seed)
        for cname in CONFIGS for pname in PROMPTS for seed in SEEDS
    ]
    print(f"{len(tasks)} generations, max {MAX_INFLIGHT} in flight", flush=True)

    def submit(t):
        cname, pname, seed = t
        mode, steps, cfg, neg, magic, ls = CONFIGS[cname]
        return gen.generate.spawn(
            prompt=PROMPTS[pname] + (MAGIC if magic else ""), mode=mode, negative_prompt=neg,
            aspect="3:4", steps=steps, cfg=cfg, seed=seed, lora_scale=ls,
        )

    # Throttle so the autoscaler doesn't start dozens of H100 containers.
    def path(t):
        return f"{OUT}/{t[0]}__{t[1]}__s{t[2]}.png"

    done = [t for t in tasks if os.path.exists(path(t))]
    print(f"resuming: {len(done)} images already on the volume", flush=True)

    class _Done:
        def __init__(self, t):
            self.t = t

        def get(self):
            return open(path(self.t), "rb").read()

    pending = [t for t in tasks if t not in done]
    inflight = [(t, _Done(t)) for t in done]
    inflight += [(t, submit(t)) for t in pending[:MAX_INFLIGHT]]
    pending = pending[MAX_INFLIGHT:]

    rows, pngs = [], {}
    i = 0
    while inflight:
        (cname, pname, seed), call = inflight.pop(0)
        png = call.get()
        if pending:
            t = pending.pop(0)
            inflight.append((t, submit(t)))
        pngs[(cname, pname, seed)] = png
        if not os.path.exists(path((cname, pname, seed))):
            with open(path((cname, pname, seed)), "wb") as f:
                f.write(png)
        s = scorer.score.remote(png, PROMPTS[pname])
        rows.append({"config": cname, "prompt": pname, "seed": seed, **s})
        i += 1
        if i % 16 == 0:
            print(f"  {i}/{len(tasks)} done", flush=True)
            outvol.commit()

    with open(f"{OUT}/scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Normalize per (prompt, seed) so easy prompts don't dominate: z-score within each group.
    import statistics as st

    groups = {}
    for r in rows:
        groups.setdefault((r["prompt"], r["seed"]), []).append(r)
    for g in groups.values():
        for k in ("pick", "hps"):
            vals = [r[k] for r in g]
            mu, sd = st.mean(vals), st.pstdev(vals) or 1.0
            for r in g:
                r[k + "_z"] = (r[k] - mu) / sd

    summary = []
    for cname in CONFIGS:
        rs = [r for r in rows if r["config"] == cname]
        summary.append({
            "config": cname,
            "pick": st.mean(r["pick"] for r in rs),
            "hps": st.mean(r["hps"] for r in rs),
            "z": st.mean((r["pick_z"] + r["hps_z"]) / 2 for r in rs),
        })
    lines = [f"{'config':28s} {'PickScore':>9s} {'HPSv2.1':>8s} {'combined_z':>10s}"]
    for fam in ("Q", "L"):
        for s in sorted((s for s in summary if s["config"].startswith(fam)), key=lambda s: -s["z"]):
            lines.append(f"{s['config']:28s} {s['pick']:9.3f} {s['hps']:8.4f} {s['z']:10.3f}")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/summary.txt", "w", encoding="utf-8") as f:
        f.write(text)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)

    # Contact sheets: one per prompt, seed 11, every config side by side.
    for pname in PROMPTS:
        ims = [pngs[(c, pname, SEEDS[0])] for c in CONFIGS]
        jpg = scorer.sheet.remote(ims, list(CONFIGS), 8)
        with open(f"{OUT}/sheet_{pname}.jpg", "wb") as f:
            f.write(jpg)
    outvol.commit()
    return text


@app.local_entrypoint()
def main():
    print(sweep.remote())
