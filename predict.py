# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import copy
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.checkpoint
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from PIL import Image
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast, logging

from flux_sliders.utils.custom_flux_pipeline import FluxPipeline
from flux_sliders.utils.lora import (
    DEFAULT_TARGET_REPLACE,
    UNET_TARGET_REPLACE_MODULE_CONV,
    LoRANetwork,
)


def concat_imgs(img_dir, save_path, direction: str = "horizontal"):
    img_path_list = list(Path(img_dir).glob("*.*"))
    img_path_list.sort()

    img_list = [Image.open(img_path) for img_path in img_path_list]

    img_sizes = np.array([v.size for v in img_list])

    if direction == "horizontal":
        width = np.sum(img_sizes[:, 0])
        height = np.max(img_sizes[:, 1])
    elif direction == "vertical":
        width = np.max(img_sizes[:, 0])
        height = np.sum(img_sizes[:, 1])
    else:
        raise ValueError(f"{direction} is not supported.")

    new_image = Image.new("RGB", (width, height), color="white")

    for i, img in enumerate(img_list):
        if i == 0:
            new_image.paste(img, (0, 0))
            continue

        if direction == "horizontal":
            x = np.sum(img_sizes[:, 0][:i])
            new_image.paste(img, (x, 0))
        elif direction == "vertical":
            y = np.sum(img_sizes[:, 1][:i])
            new_image.paste(img, (0, y))

    new_image.save(save_path)


def image_grid(imgs):
    """Load and show images in a grid from a list of paths"""
    count = len(imgs)
    plt.figure(figsize=(11, 18))
    for ix, path in enumerate(imgs):
        plt.subplots_adjust(bottom=0.3, right=0.8, top=0.5)
        ax = plt.subplot(3, 5, ix + 1)
        ax.axis("off")
        plt.imshow(path)
    plt.tight_layout()


logging.set_verbosity_warning()

from diffusers import logging

logging.set_verbosity_error()
modules = DEFAULT_TARGET_REPLACE

pretrained_model_name_or_path = "models/FLUX.1-dev"
weight_dtype = torch.bfloat16

device = "cuda:0"
max_train_steps = 1000
num_inference_steps = 30
guidance_scale = 3.5
max_sequence_length = 512
height = width = 512
if "schnell" in pretrained_model_name_or_path:
    num_inference_steps = 4
    guidance_scale = 0
    max_sequence_length = 256

# timestep weighting
weighting_scheme = "none"  # ["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"]
logit_mean = 0.0
logit_std = 1.0
mode_scale = 1.29
bsz = 1
training_eta = 1
# optimizer params
lr = 0.002

output_dir = "./outputs/models/fluxsliders"
if "schnell" in pretrained_model_name_or_path:
    output_dir = output_dir + "/schnell/"
else:
    output_dir = output_dir + "/dev/"
os.makedirs(output_dir, exist_ok=True)

target_prompt = "picture of a person"
positive_prompt = "photo of a person, smiling, happy"
negative_prompt = "photo of a person, frowning"

slider_name = "person-smiling"

# lora params
alpha = 1
rank = 16
train_method = "xattn"
num_sliders = 1

# training params
batchsize = 1
eta = 2


def unwrap_model(model):
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)
    while isinstance(model, options):
        model = model.module
    return model


# Function to log gradients
def log_gradients(named_parameters):
    grad_dict = defaultdict(lambda: defaultdict(float))
    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            grad_dict[name]["mean"] = param.grad.abs().mean().item()
            grad_dict[name]["std"] = param.grad.std().item()
            grad_dict[name]["max"] = param.grad.abs().max().item()
            grad_dict[name]["min"] = param.grad.abs().min().item()
    return grad_dict


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, device_map=device
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel

    if model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    raise ValueError(f"{model_class} is not supported.")


def load_text_encoders(
    pretrained_model_name_or_path, class_one, class_two, weight_dtype
):
    text_encoder_one = class_one.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=weight_dtype,
        device_map=device,
    )
    text_encoder_two = class_two.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        torch_dtype=weight_dtype,
        device_map=device,
    )
    return text_encoder_one, text_encoder_two


def plot_labeled_images(images, labels, idx, timestamp: str):
    # Determine the number of images
    n = len(images)

    # Create a new figure with a single row
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    # If there's only one image, axes will be a single object, not an array
    if n == 1:
        axes = [axes]

    # Plot each image
    for i, (img, label) in enumerate(zip(images, labels)):
        # Convert PIL image to numpy array
        img_array = np.array(img)

        # Display the image
        axes[i].imshow(img_array)
        axes[i].axis("off")  # Turn off axis

        # Set the title (label) for the image
        axes[i].set_title(label)

    # Adjust the layout and display the plot
    # plt.tight_layout()
    output_dir = Path("tmp") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(output_dir) / f"vis_{idx}.jpg"
    plt.savefig(str(save_path))


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(
        device=device, dtype=dtype
    )
    text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def compute_text_embeddings(prompt, text_encoders, tokenizers):
    device = text_encoders[0].device
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length=max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def get_sigmas(timesteps, n_dim=4, device="cuda:0", dtype=torch.bfloat16):
    sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, "same")


if __name__ == "__main__":
    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        torch_dtype=weight_dtype,
        device_map=device,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        torch_dtype=weight_dtype,
        device_map=device,
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="scheduler",
        torch_dtype=weight_dtype,
        device_map=device,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path,
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    # Load the text encoders
    text_encoder_one, text_encoder_two = load_text_encoders(
        pretrained_model_name_or_path,
        text_encoder_cls_one,
        text_encoder_cls_two,
        weight_dtype,
    )

    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=weight_dtype,
        device_map=int(device.split(":")[1]),
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype
    )

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    vae.to(device)
    transformer.to(device)
    text_encoder_one.to(device)
    text_encoder_two.to(device)
    print("Loaded Models")

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
            [target_prompt, positive_prompt, negative_prompt], text_encoders, tokenizers
        )
        (
            target_prompt_embeds,
            positive_prompt_embeds,
            negative_prompt_embeds,
        ) = prompt_embeds.chunk(3)
        (
            target_pooled_prompt_embeds,
            positive_pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pooled_prompt_embeds.chunk(3)
        target_text_ids, positive_text_ids, negative_text_ids = text_ids.chunk(3)

    networks = {}
    params = []
    modules = DEFAULT_TARGET_REPLACE
    modules += UNET_TARGET_REPLACE_MODULE_CONV
    for i in range(num_sliders):
        networks[i] = LoRANetwork(
            transformer,
            rank=rank,
            multiplier=1.0,
            alpha=alpha,
            train_method=train_method,
        ).to(device, dtype=weight_dtype)
        params.extend(networks[i].prepare_optimizer_params())

    pipe = FluxPipeline(
        noise_scheduler,
        vae,
        text_encoder_one,
        tokenizer_one,
        text_encoder_two,
        tokenizer_two,
        transformer,
    )
    pipe.set_progress_bar_config(disable=True)

    slider_path = "outputs/fluxsliders/dev/flux-person-zoom"
    print("Loading...")
    for i in range(num_sliders):
        slider_full_path = f"{slider_path}/slider_{i}.pt"
        print(f"Loading {slider_full_path}")
        networks[i].load_state_dict(torch.load(slider_full_path))
    print("Done.")

    print("Inference")
    target_prompt = "woman with red hair, playing chess at the park, bomb going off in the background"
    prompts = [target_prompt]

    # LoRA weights/scale to test
    slider_scales = [-5, -2.5, 0, 2.5, 5]

    num_images = len(slider_scales)

    seeds = [42 + i for i in range(num_images)]

    timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")
    for net in networks:
        print(f"Slider {net}")
        for idx in range(num_images):
            slider_images = []
            seed = seeds[idx]
            for slider_scale in slider_scales:
                networks[net].set_lora_slider(scale=slider_scale)
                with torch.no_grad():
                    image = pipe(
                        target_prompt,
                        height=height,
                        width=width,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        max_sequence_length=max_sequence_length,
                        num_images_per_prompt=1,
                        generator=torch.Generator().manual_seed(seed),
                        from_timestep=0,
                        till_timestep=None,
                        output_type="pil",
                        network=networks[net],
                        skip_slider_timestep_till=1,  # this will skip adding the slider on the first step of generation ('1' will skip first 2 steps)
                    )
                slider_images.append(image.images[0])
            plot_labeled_images(slider_images, slider_scales, idx, timestamp)

    img_dir = Path("tmp") / timestamp
    concat_imgs(img_dir, img_dir / "results.jpg", direction="vertical")
