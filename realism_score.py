"""Zero-shot 'looks like a real photo' probe with CLIP ViT-H-14 (LAION-2B).

realism = P(real-photo captions) vs P(AI / render / airbrushed captions), averaged.
A rough signal only; human eyes are the final judge.

  modal run realism_score.py
"""
import collections
import os

import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", "open_clip_torch", "huggingface_hub", "pillow"
).env({"HF_HOME": "/cache"})
cache = modal.Volume.from_name("qwen-image-hf-cache", create_if_missing=True)
outvol = modal.Volume.from_name("qwen-tune-out", create_if_missing=True)
app = modal.App("qwen-realism-score", image=image)

REAL = [
    "an unedited candid photo taken with a smartphone",
    "a real amateur photograph of a person",
    "a raw unretouched photo with natural skin texture",
]
FAKE = [
    "an AI-generated image",
    "a 3D render of a person",
    "a heavily airbrushed and retouched glamour photo with plastic skin",
    "a digital painting",
    "a wax figure",
]


@app.function(gpu="A10G", volumes={"/cache": cache, "/out": outvol}, timeout=1800)
def score() -> str:
    import open_clip
    import torch
    from PIL import Image

    model, _, pre = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", cache_dir="/cache/open_clip")
    model = model.eval().half().cuda()
    tok = open_clip.get_tokenizer("ViT-H-14")
    with torch.no_grad():
        t = model.encode_text(tok(REAL + FAKE).cuda(), normalize=True)
    cache.commit()

    files = []
    for root in ("/out/realism", "/out"):
        for f in sorted(os.listdir(root)):
            if f.endswith(".png") and (root == "/out/realism" or f.split("__")[1] in ("portrait", "elder")):
                files.append(os.path.join(root, f))

    res = {}
    for fn in files:
        x = pre(Image.open(fn).convert("RGB")).unsqueeze(0).half().cuda()
        with torch.no_grad():
            i = model.encode_image(x, normalize=True)
            p = (100 * i @ t.T).softmax(-1)[0]
        res[fn] = p[: len(REAL)].sum().item()

    groups = collections.defaultdict(list)
    for fn, r in res.items():
        parts = os.path.basename(fn)[:-4].split("__")
        if "/realism/" in fn:  # subj__variant__te__seed
            groups[f"realism  {parts[1]:22s} TE={parts[2]}"].append(r)
        else:                  # config__prompt__seed
            groups[f"sweep    {parts[0]:22s}"].append(r)
    lines = [f"{'group':48s} {'P(real photo)':>13s}  n"]
    for k, v in sorted(groups.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        lines.append(f"{k:48s} {sum(v) / len(v):13.3f}  {len(v)}")
    text = "\n".join(lines)
    with open("/out/realism/realism_scores.txt", "w") as f:
        f.write(text)
    outvol.commit()
    return text


@app.local_entrypoint()
def main():
    print(score.remote())
