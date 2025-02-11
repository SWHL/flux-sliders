# -*- encoding: utf-8 -*-
import copy
import random
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.checkpoint
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

from .utils.custom_flux_pipeline import FluxPipeline
from .utils.lora import LoRANetwork
from .utils.utils import get_cur_timestamp, mkdir

root_dir = Path(__file__).resolve().parent
DEFAULT_TEXT_SLIDER_CONFIG = root_dir / "config" / "default_text_slider_cfg.yaml"


@dataclass
class PromptInputDataTriplet:
    target_prompt: str
    target_prompt_embeds: torch.Tensor
    target_pooled_prompt_embeds: torch.Tensor
    target_text_ids: torch.Tensor

    positive_prompt: str
    positive_prompt_embeds: torch.Tensor
    positive_pooled_prompt_embeds: torch.Tensor
    positive_text_ids: torch.Tensor

    negative_prompt: str
    negative_prompt_embeds: torch.Tensor
    negative_pooled_prompt_embeds: torch.Tensor
    negative_text_ids: torch.Tensor


class FLUXTextSliders:
    def __init__(self, config_file: Union[str, Path] = DEFAULT_TEXT_SLIDER_CONFIG):
        print(f"Loading config from {config_file}")

        self.cfg = OmegaConf.load(config_file)
        self.weight_dtype = torch.bfloat16

        print("Load the tokenizers")
        tokenizer_one = CLIPTokenizer.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="tokenizer",
            torch_dtype=self.weight_dtype,
            device_map=self.cfg.device,
        )
        tokenizer_two = T5TokenizerFast.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            torch_dtype=self.weight_dtype,
            device_map=self.cfg.device,
        )

        print("Load scheduler and models")
        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weight_dtype,
            device_map=self.cfg.device,
        )
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)

        text_encoder_cls_one = self.import_model_class_from_model_name_or_path(
            self.cfg.pretrained_model_name_or_path, device=self.cfg.device
        )
        text_encoder_cls_two = self.import_model_class_from_model_name_or_path(
            self.cfg.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            device=self.cfg.device,
        )
        print("Load the text encoders")
        text_encoder_one, text_encoder_two = self.load_text_encoders(
            self.cfg.pretrained_model_name_or_path,
            text_encoder_cls_one,
            text_encoder_cls_two,
            self.weight_dtype,
            device=self.cfg.device,
        )

        print("Load VAE")
        vae = AutoencoderKL.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=self.weight_dtype,
            device_map=int(self.cfg.device.split(":")[1]),
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=self.weight_dtype,
        )

        # We only train the additional adapter LoRA layers
        transformer.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)

        vae.to(self.cfg.device)
        transformer.to(self.cfg.device)
        text_encoder_one.to(self.cfg.device)
        text_encoder_two.to(self.cfg.device)
        print("Loaded Models")

        self.vae = vae
        self.transformer = transformer
        self.tokenizers = [tokenizer_one, tokenizer_two]
        self.text_encoders = [text_encoder_one, text_encoder_two]

        self.pipe = FluxPipeline(
            noise_scheduler,
            vae,
            text_encoder_one,
            tokenizer_one,
            text_encoder_two,
            tokenizer_two,
            transformer,
        )

        save_name = f"flux-{self.cfg.slider_name}"
        self.save_dir = Path(self.cfg.output_dir) / save_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.export_cfg()

    def train(self):
        promt_input_data = self.get_input()

        self.init_lora()
        optimizer = AdamW(self.params, lr=self.cfg.lr)
        optimizer.zero_grad()

        lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.cfg.lr_warmup_steps,
            num_training_steps=self.cfg.max_train_steps,
            num_cycles=self.cfg.lr_num_cycles,
            power=self.cfg.lr_power,
        )

        progress_bar = tqdm(range(0, self.cfg.max_train_steps), desc="Steps")

        losses = {}
        for step in range(self.cfg.max_train_steps):
            prompt_data = random.choice(promt_input_data)

            u = compute_density_for_timestep_sampling(
                weighting_scheme=self.cfg.weighting_scheme,
                batch_size=self.cfg.bsz,
                logit_mean=self.cfg.logit_mean,
                logit_std=self.cfg.logit_std,
                mode_scale=self.cfg.mode_scale,
            )
            indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler_copy.timesteps[indices].to(
                device=self.cfg.device
            )

            # get initial latents or x_t to train
            timestep_to_infer = (
                (
                    indices[0]
                    * (
                        self.cfg.num_inference_steps
                        / self.noise_scheduler_copy.config.num_train_timesteps
                    )
                )
                .long()
                .item()
            )

            with torch.no_grad():
                packed_noisy_model_input = self.pipe(
                    prompt_data.target_prompt,
                    height=self.cfg.height,
                    width=self.cfg.width,
                    guidance_scale=self.cfg.guidance_scale,
                    num_inference_steps=self.cfg.num_inference_steps,
                    max_sequence_length=self.cfg.max_sequence_length,
                    num_images_per_prompt=self.cfg.bsz,
                    generator=None,
                    from_timestep=0,
                    till_timestep=timestep_to_infer,
                    output_type="latent",
                )
                vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels))

                if step == 0:
                    model_input = FluxPipeline._unpack_latents(
                        packed_noisy_model_input,
                        height=self.cfg.height,
                        width=self.cfg.width,
                        vae_scale_factor=vae_scale_factor,
                    )

            latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                model_input.shape[0],
                model_input.shape[2],
                model_input.shape[3],
                self.cfg.device,
                self.weight_dtype,
            )

            # handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.tensor(
                    [self.cfg.guidance_scale], device=self.cfg.device
                )
                guidance = guidance.expand(model_input.shape[0])
            else:
                guidance = None

            with ExitStack() as stack:
                stack.enter_context(self.network)

                model_pred = self.transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=prompt_data.target_pooled_prompt_embeds,
                    encoder_hidden_states=prompt_data.target_prompt_embeds,
                    txt_ids=prompt_data.target_text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

            model_pred = FluxPipeline._unpack_latents(
                model_pred,
                height=int(model_input.shape[2] * vae_scale_factor / 2),
                width=int(model_input.shape[3] * vae_scale_factor / 2),
                vae_scale_factor=vae_scale_factor,
            )

            with torch.no_grad():
                target_pred = self.get_transformer_output(
                    prompt_data.target_prompt_embeds,
                    prompt_data.target_pooled_prompt_embeds,
                    prompt_data.target_text_ids,
                    timesteps,
                    packed_noisy_model_input,
                    vae_scale_factor,
                    model_input,
                    latent_image_ids,
                    guidance,
                )
                positive_pred = self.get_transformer_output(
                    prompt_data.positive_prompt_embeds,
                    prompt_data.positive_pooled_prompt_embeds,
                    prompt_data.positive_text_ids,
                    timesteps,
                    packed_noisy_model_input,
                    vae_scale_factor,
                    model_input,
                    latent_image_ids,
                    guidance,
                )
                negative_pred = self.get_transformer_output(
                    prompt_data.negative_prompt_embeds,
                    prompt_data.negative_pooled_prompt_embeds,
                    prompt_data.negative_text_ids,
                    timesteps,
                    packed_noisy_model_input,
                    vae_scale_factor,
                    model_input,
                    latent_image_ids,
                    guidance,
                )

                gt_pred = target_pred + self.cfg.eta * (positive_pred - negative_pred)
                gt_pred = (gt_pred / gt_pred.norm()) * positive_pred.norm()

            concept_loss = torch.mean(
                ((model_pred.float() - gt_pred.float()) ** 2).reshape(
                    gt_pred.shape[0], -1
                ),
                1,
            )
            concept_loss = concept_loss.mean()

            concept_loss.backward()
            losses["concept"] = losses.get("concept", []) + [concept_loss.item()]

            logs = {
                "concept loss": losses["concept"][-1],
                "lr": lr_scheduler.get_last_lr()[0],
            }

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

            if step % self.cfg.sample_every == 0:
                self.save_lora_weights(name_suffix=f"{step:06d}")
                self.inference(prompt_data.target_prompt, step=f"{step:06d}")

        self.plot_history(losses)
        self.save_lora_weights(name_suffix="latest")

    def get_input(self) -> List[PromptInputDataTriplet]:
        prompt_inputs = []
        for v in self.cfg.prompt.values():
            with torch.no_grad():
                (
                    prompt_embeds,
                    pooled_prompt_embeds,
                    text_ids,
                ) = self.compute_text_embeddings(
                    [
                        v.target_prompt,
                        v.positive_prompt,
                        v.negative_prompt,
                    ],
                    self.text_encoders,
                    self.tokenizers,
                    self.cfg.max_sequence_length,
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

                target_text_ids, positive_text_ids, negative_text_ids = text_ids.chunk(
                    3
                )
            prompt_inputs.append(
                PromptInputDataTriplet(
                    v.target_prompt,
                    target_prompt_embeds,
                    target_pooled_prompt_embeds,
                    target_text_ids,
                    v.positive_prompt,
                    positive_prompt_embeds,
                    positive_pooled_prompt_embeds,
                    positive_text_ids,
                    v.negative_prompt,
                    negative_prompt_embeds,
                    negative_pooled_prompt_embeds,
                    negative_text_ids,
                )
            )
        return prompt_inputs

    def save_lora_weights(self, name_suffix: str):
        save_weight_dir = self.save_dir / "weights"
        save_weight_dir.mkdir(parents=True, exist_ok=True)
        save_weight_path = (
            save_weight_dir / f"flux-{self.cfg.slider_name}_{name_suffix}.safetensors"
        )
        self.network.save_weights(str(save_weight_path), dtype=self.weight_dtype)

    def init_lora(self):
        params = []
        network = LoRANetwork(
            self.transformer,
            rank=self.cfg.rank,
            multiplier=1.0,
            alpha=self.cfg.alpha,
            train_method=self.cfg.train_method,
            save_dir=self.save_dir,
        ).to(self.cfg.device, dtype=self.weight_dtype)
        params.extend(network.prepare_optimizer_params())
        self.network = network
        self.params = params

    def load_lora_weights(self, slider_path: Union[str, Path], slider_idx: int = 0):
        slider_path = Path(slider_path)
        if slider_path.suffix == ".safetensors":
            state_dict = load_file(slider_path)
        else:
            state_dict = torch.load(slider_path)
        self.network.load_state_dict(state_dict)

    def inference(
        self,
        target_prompt: str,
        step: Optional[int] = None,
        num_images: int = 1,
        slider_scales: Tuple[float] = (-5, -2.5, 0, 2.5, 5),
        seed: Optional[int] = None,
    ):
        save_vis_dir = self.save_dir / "sample"
        mkdir(save_vis_dir)

        save_single_dir = save_vis_dir / "single"
        save_single_dir.mkdir(parents=True, exist_ok=True)

        if seed is None:
            seeds = [random.randint(0, 2**15) for _ in range(num_images)]
        else:
            seeds = [seed] * num_images

        for idx in range(num_images):
            slider_images = []
            seed = seeds[idx]

            time_stamp = get_cur_timestamp()
            for slider_scale in slider_scales:
                self.network.set_lora_slider(scale=slider_scale)
                with torch.no_grad():
                    image = self.pipe(
                        target_prompt,
                        height=self.cfg.height,
                        width=self.cfg.width,
                        guidance_scale=self.cfg.guidance_scale,
                        num_inference_steps=self.cfg.num_inference_steps,
                        max_sequence_length=self.cfg.max_sequence_length,
                        num_images_per_prompt=1,
                        generator=torch.Generator().manual_seed(seed),
                        from_timestep=0,
                        till_timestep=None,
                        output_type="pil",
                        network=self.network,
                        skip_slider_timestep_till=0,  # this will skip adding the slider on the first step of generation ('1' will skip first 2 steps)
                    )
                gen_img = image.images[0]
                slider_images.append(gen_img)
                gen_img.save(
                    save_single_dir / f"{time_stamp}_img_{idx}_scale_{slider_scale}.jpg"
                )

            save_img_path = save_vis_dir / f"{time_stamp}_img_{idx}.jpg"

            if step is not None:
                save_img_path = save_vis_dir / f"{step}_img_{idx}.jpg"

            self.plot_labeled_images(slider_images, slider_scales, idx, save_img_path)
        return slider_images[0]

    def get_transformer_output(
        self,
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
        timesteps,
        packed_noisy_model_input,
        vae_scale_factor,
        model_input,
        latent_image_ids,
        guidance,
    ):
        target_pred = self.transformer(
            hidden_states=packed_noisy_model_input,
            timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]
        target_pred = FluxPipeline._unpack_latents(
            target_pred,
            height=int(model_input.shape[2] * vae_scale_factor / 2),
            width=int(model_input.shape[3] * vae_scale_factor / 2),
            vae_scale_factor=vae_scale_factor,
        )
        return target_pred

    @staticmethod
    def image_grid(imgs):
        """Load and show images in a grid from a list of paths"""
        plt.figure(figsize=(11, 18))
        for ix, path in enumerate(imgs):
            plt.subplots_adjust(bottom=0.3, right=0.8, top=0.5)
            ax = plt.subplot(3, 5, ix + 1)
            ax.axis("off")
            plt.imshow(path)
        plt.tight_layout()

    @staticmethod
    def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str,
        subfolder: str = "text_encoder",
        device: str = "cuda:0",
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

    @staticmethod
    def load_text_encoders(
        pretrained_model_name_or_path, class_one, class_two, weight_dtype, device
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

    def plot_labeled_images(self, images, labels, idx, save_img_path: Path):
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1:
            axes = [axes]

        for i, (img, label) in enumerate(zip(images, labels)):
            img_array = np.array(img)

            axes[i].imshow(img_array)
            axes[i].axis("off")
            axes[i].set_title(label)

        plt.tight_layout(w_pad=5)
        plt.savefig(str(save_img_path))
        print(save_img_path)

    @staticmethod
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

    @staticmethod
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
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return prompt_embeds

    @staticmethod
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

        prompt_embeds = text_encoder(
            text_input_ids.to(device), output_hidden_states=False
        )

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
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

        pooled_prompt_embeds = self._encode_prompt_with_clip(
            text_encoder=text_encoders[0],
            tokenizer=tokenizers[0],
            prompt=prompt,
            device=device if device is not None else text_encoders[0].device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
        )

        prompt_embeds = self._encode_prompt_with_t5(
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

    def compute_text_embeddings(
        self, prompt, text_encoders, tokenizers, max_sequence_length
    ):
        device = text_encoders[0].device
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                text_encoders,
                tokenizers,
                prompt,
                max_sequence_length=max_sequence_length,
            )
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
            text_ids = text_ids.to(device)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    def plot_history(self, history):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
        ax1.plot(history["concept"])
        ax1.set_title("Concept Loss")
        ax2.plot(self.movingaverage(history["concept"], 10))
        ax2.set_title("Moving Average Concept Loss")
        plt.tight_layout()

        save_path = self.save_dir / "loss.png"
        plt.savefig(str(save_path))

    @staticmethod
    def movingaverage(interval, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(interval, window, "same")

    def export_cfg(self):
        save_cfg_yaml = self.save_dir / "config.yaml"
        if save_cfg_yaml.exists():
            print(f"{save_cfg_yaml} already exists. Skipping saving config.")
            return

        with open(save_cfg_yaml, "w", encoding="utf-8") as f:
            OmegaConf.save(self.cfg, f)
        print("Config saved")
