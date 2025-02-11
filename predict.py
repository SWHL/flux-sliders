# -*- encoding: utf-8 -*-
from flux_sliders.text_sliders import FLUXTextSliders

config_file = "config/smile_sliders.yaml"
model = FLUXTextSliders(config_file)

model.init_lora()
model.load_lora_weights("outputs/flux-person-smiling/weights/slider_0.safetensors")

prompt = "picture of a man"
model.inference(prompt, num_images=1)
