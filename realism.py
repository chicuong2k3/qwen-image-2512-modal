"""Realism experiment: why do people look 'AI'?

Compares, on the same seeds:
  - text encoder: abliterated (sci4ai) vs original Qwen-Image-2512 encoder
  - prompt style: "glossy" (85mm/8k/masterpiece tags) vs "snapshot" (plain natural
    description, casual phone photo, as in the official model card) vs snapshot + magic suffix
  - base 50 steps cfg 4 (official neg prompt) vs Lightning 4 steps

  modal run --detach realism.py
  modal volume get qwen-tune-out realism tune_out
"""
import io
import math
import os

import modal

BASE_REPO = "Qwen/Qwen-Image-2512"
ABLIT_TE = "sci4ai/Qwen2.5-VL-7B-Instruct-Ablit"
LORA_REPO = "lightx2v/Qwen-Image-2512-Lightning"
LORA_4 = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
OFFICIAL_NEG = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，"
    "画面具有AI感。构图混乱。文字模糊，扭曲。"
)
MAGIC = ", Ultra HD, 4K, cinematic composition."

SUBJECTS = {
    "woman": {
        "glossy": "Photorealistic portrait photo of a young Vietnamese woman in her mid-20s, natural makeup, long black hair, soft smile, beige linen shirt, Hanoi street at golden hour, shallow depth of field, 85mm lens f/1.8, natural skin texture with pores, soft rim light, shot on Canon EOS R5, ultra detailed, 8k, masterpiece",
        "snapshot": "A casual smartphone photo of a Vietnamese woman in her mid-20s sitting at a small sidewalk cafe in Hanoi. She has long black hair slightly messy from the wind, no makeup, visible skin texture with a few small blemishes and fine baby hairs, and an unposed half-smile while looking slightly off camera. She wears a plain beige linen shirt. Behind her are parked motorbikes, a street vendor and shop signs, in focus. Ordinary overcast daylight, no studio lighting, slightly imperfect framing, like a photo a friend took on an iPhone.",
    },
    "elder": {
        "glossy": "Candid street photograph of an elderly man with a weathered face and white stubble drinking coffee at a small plastic table in a Saigon alley, morning light, natural skin texture, film grain, 35mm lens, documentary style, ultra realistic, 8k, masterpiece",
        "snapshot": "A phone snapshot of a Vietnamese man around 70 years old sitting on a low blue plastic stool in a narrow Saigon alley, holding a glass of iced coffee. Deep wrinkles, sun spots and age spots on his face and hands, uneven white stubble, slightly yellowed teeth as he laughs at something off camera. Faded checkered shirt. Behind him: worn concrete walls, electric wires, a parked scooter. Flat morning light, ordinary colors, no dramatic lighting, unposed everyday moment.",
    },
    "office": {
        "glossy": "Photorealistic portrait of a 30-year-old Asian man in an office, white shirt, confident smile, modern office background with bokeh, studio lighting, 85mm, sharp focus, ultra detailed skin, 8k, masterpiece",
        "snapshot": "A slightly tilted selfie taken on a phone by a tired 30-year-old Vietnamese office worker at his desk late in the afternoon. Short black hair a bit oily, faint dark circles under his eyes, light stubble, a small smile. Wrinkled light-blue shirt. Behind him: a cluttered desk with a monitor, sticky notes, a half-empty plastic water bottle, fluorescent ceiling lights. Typical indoor office lighting, slightly noisy phone camera image.",
    },
}
SEEDS = [11, 22]
SIZE = (1104, 1472)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers>=5", "diffusers>=0.36", "accelerate", "peft",
                 "safetensors", "sentencepiece", "huggingface_hub", "pillow")
    .env({"HF_HOME": "/cache"})
)
cache = modal.Volume.from_name("qwen-image-hf-cache", create_if_missing=True)
outvol = modal.Volume.from_name("qwen-tune-out", create_if_missing=True)
app = modal.App("qwen-image-realism", image=image)


@app.cls(gpu="H100", volumes={"/cache": cache, "/out": outvol}, timeout=2 * 3600)
class Gen:
    te: str = modal.parameter()  # "ablit" or "orig"

    @modal.enter()
    def load(self):
        import torch
        from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
        from transformers import Qwen2_5_VLForConditionalGeneration

        kw = {}
        if self.te == "ablit":
            kw["text_encoder"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                ABLIT_TE, torch_dtype=torch.bfloat16)
        pipe = DiffusionPipeline.from_pretrained(BASE_REPO, torch_dtype=torch.bfloat16, **kw)
        pipe.load_lora_weights(LORA_REPO, weight_name=LORA_4, adapter_name="l4")
        pipe.disable_lora()
        self.pipe = pipe.to("cuda")
        self.sched_base = pipe.scheduler
        self.sched_fast = FlowMatchEulerDiscreteScheduler.from_config({
            "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False,
            "max_image_seq_len": 8192, "max_shift": math.log(3), "num_train_timesteps": 1000,
            "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False,
            "time_shift_type": "exponential", "use_beta_sigmas": False,
            "use_dynamic_shifting": True, "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        })
        cache.commit()

    @modal.method()
    def run(self) -> list[str]:
        import torch

        os.makedirs("/out/realism", exist_ok=True)
        variants = [
            ("Q50_glossy", "glossy", False, False),
            ("Q50_snapshot", "snapshot", False, False),
            ("Q50_snapshot_magic", "snapshot", True, False),
            ("L4_snapshot", "snapshot", False, True),
        ]
        written = []
        for subj, styles in SUBJECTS.items():
            for vname, style, magic, light in variants:
                for seed in SEEDS:
                    fn = f"/out/realism/{subj}__{vname}__{self.te}__s{seed}.png"
                    if os.path.exists(fn):
                        continue
                    if light:
                        self.pipe.enable_lora()
                        self.pipe.set_adapters(["l4"], adapter_weights=[1.0])
                        self.pipe.scheduler = self.sched_fast
                        steps, cfg = 4, 1.0
                    else:
                        self.pipe.disable_lora()
                        self.pipe.scheduler = self.sched_base
                        steps, cfg = 50, 4.0
                    img = self.pipe(
                        prompt=styles[style] + (MAGIC if magic else ""),
                        negative_prompt=OFFICIAL_NEG, width=SIZE[0], height=SIZE[1],
                        num_inference_steps=steps, true_cfg_scale=cfg,
                        generator=torch.Generator("cuda").manual_seed(seed),
                    ).images[0]
                    img.save(fn)
                    written.append(fn)
                    print("wrote", fn, flush=True)
                outvol.commit()
        return written


@app.function(volumes={"/out": outvol}, timeout=600)
def sheets():
    """One sheet per subject: rows = variants x encoder, cols = seeds."""
    from PIL import Image, ImageDraw

    outvol.reload()
    tw, th = 552, 736
    variants = ["Q50_glossy", "Q50_snapshot", "Q50_snapshot_magic", "L4_snapshot"]
    for subj in SUBJECTS:
        cells = [(v, te, s) for v in variants for te in ("ablit", "orig") for s in SEEDS]
        cols = 4
        rows = (len(cells) + cols - 1) // cols
        out = Image.new("RGB", (cols * tw, rows * (th + 26)), "white")
        d = ImageDraw.Draw(out)
        for i, (v, te, s) in enumerate(cells):
            fn = f"/out/realism/{subj}__{v}__{te}__s{s}.png"
            x, y = (i % cols) * tw, (i // cols) * (th + 26)
            d.text((x + 4, y + 6), f"{v} | TE={te} | seed {s}", fill="black")
            if os.path.exists(fn):
                out.paste(Image.open(fn).convert("RGB").resize((tw, th)), (x, y + 26))
        out.save(f"/out/realism/sheet_{subj}.jpg", quality=88)
    outvol.commit()


@app.function(timeout=3 * 3600)
def orchestrate():
    calls = [Gen(te=te).run.spawn() for te in ("ablit", "orig")]
    for c in calls:
        print(len(c.get()), "images", flush=True)
    sheets.remote()
    print("done", flush=True)


@app.local_entrypoint()
def main():
    # Runs fully server-side so a local network drop can't kill it (use with --detach).
    orchestrate.remote()
