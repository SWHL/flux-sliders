# -*- encoding: utf-8 -*-
from flux_sliders.text_sliders import FLUXTextSliders

config_file = "config/smile_slider.yaml"
model = FLUXTextSliders(config_file)
model.train()
model.inference("picture of a man")
