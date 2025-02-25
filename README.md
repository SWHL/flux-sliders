## FLUX Sliders

修改自<https://github.com/rohitgandikota/sliders>

⚠️注意：代码仍在快速迭代中，欢迎加入。

当前代码训练所得模型：

- 可以直接放在**Comfy/models/loras**下被加载使用。但是`slider scale`暂时不知如何插入进去。
- 可以直接使用diffusers库来推理

### TODO

- [x] 支持diffusers直接推理
- [x] 支持ComfyUI推理
- [x] 多个slider lora融合
- [ ] slider scale与原本lora scale的关系探究？

### 推荐显存

~80G

可以自行将FLUX模型替换为量化后的FLUX模型

### 安装环境

```bash
conda env create -f environment.yml
```

### 下载flux-dev模型

```bash
huggingface-cli login
huggingface-cli download --resume-download black-forest-labs/FLUX.1-dev --local-dir models/FLUX.1-dev
```

### 训练

直接执行以下命令，会训练人物由皱眉到微笑的sliders。

```bash
python train_text_sliders.py
```

### 推理

单个slider LoRA使用

```python
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
```

多个sliders LoRA结合使用

```python
from datetime import datetime
from pathlib import Path

import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("models/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda:1")

hair_lora_path = "flux-hair_sliders_latest_multiplied_1.0.safetensors"
smile_lora_path = "flux-smile_sliders_latest_multiplied_1.0.safetensors"

pipe.load_lora_weights(hair_lora_path, adapter_name="hair")
pipe.load_lora_weights(smile_lora_path, adapter_name="smile")

pipe.set_adapters(["hair", "smile"], adapter_weights=[1.0, 1.0])

time_stamp = datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")
save_dir = Path("outputs/tmp") / time_stamp
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
```

### 推理结果

![smiling_demo1](assets/smile_sliders_demo1.jpg)
![smiling_demo2](assets/smile_sliders_demo2.jpg)

更多的Sliders结果：

- [smile-sliders-flux-1d-lora](https://civitai.com/models/1230985/smile-sliders-flux-1d-lora)
- [age-sliders-flux-1d-lora](https://civitai.com/models/1242004/age-sliders-flux-1d-lora)
- [hair-sliders-flux-1d-lora](https://civitai.com/models/1245348/hair-sliders-flux-1d-lora)
