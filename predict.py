# -*- encoding: utf-8 -*-
from datetime import datetime
from pathlib import Path

import torch
from diffusers import FluxPipeline

lora_path = "flux-age_sliders_latest.safetensors"
pipe = FluxPipeline.from_pretrained("models/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.load_lora_weights(lora_path)

time_stamp = datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")
save_dir = Path("outputs") / time_stamp
save_dir.mkdir(parents=True, exist_ok=True)

scales = (-5, -2.5, 0, 2.5, 5)
prompt = "female person"

for scale in scales:
    out = pipe(
        prompt=prompt,
        guidance_scale=3.5,
        height=512,
        width=512,
        num_inference_steps=25,
        joint_attention_kwargs={"scale": scale * 1 / 16},
        generator=torch.Generator().manual_seed(42),
    ).images[0]

    save_img_path = save_dir / f"{time_stamp}_scale_{scale}.jpg"
    out.save(save_img_path)
