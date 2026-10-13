"""Generate 5 Re:Zero images against the DEPLOYED app and time each call."""
import time

import modal

Cls = modal.Cls.from_name("qwen-image-2512-ablit", "QwenImage")
svc = Cls()

prompts = {
    "subaru": "Anime illustration of Natsuki Subaru from Re:Zero, black short hair, sharp eyes, green and white tracksuit jacket, grinning determinedly, fantasy medieval city street at dusk, dynamic angle, clean lineart, cel shading, highly detailed, masterpiece",
    "rem_ram": "Anime key visual of the twin maids Rem and Ram from Re:Zero standing back to back, blue hair and pink hair, maid outfits, mansion garden, golden hour, vibrant colors, cel shading, highly detailed, masterpiece",
    "beatrice": "Anime illustration of Beatrice from Re:Zero, blonde twin drills, blue butterfly-shaped eyes, ornate pink-and-white dress, sitting in a huge magical library filled with floating books, warm candlelight, whimsical, detailed lineart, masterpiece",
    "satella": "Anime illustration of the Witch of Envy Satella from Re:Zero, long silver hair, dark flowing dress, ominous black mist swirling around her, dark sky with a red moon, dramatic lighting, haunting beauty, cel shading, highly detailed, masterpiece",
    "felt": "Anime illustration of Felt from Re:Zero, short blonde hair, red eyes, ragged street clothes, cheeky smile, rooftops of a fantasy slum city at sunset, dynamic parkour pose, vivid colors, cel shading, highly detailed, masterpiece",
}

times = []
for name, p in prompts.items():
    t = time.time()
    png = svc.generate.remote(prompt=p, aspect="3:4")
    dt = time.time() - t
    times.append(dt)
    open(f"rezero_{name}.png", "wb").write(png)
    print(f"{name}: {dt:.1f}s", flush=True)

print("times:", [round(x, 1) for x in times])
