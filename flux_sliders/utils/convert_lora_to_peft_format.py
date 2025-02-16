# Modified from https://github.com/ostris/ai-toolkit/blob/bd8d7dc0817590cfc22ff97bb2dbe66034ab48e7/scripts/convert_lora_to_peft_format.py
from pathlib import Path
from typing import Optional, Union

from safetensors.torch import load_file, save_file


class ConvertLoRAToPEFT:
    def __init__(self):
        self.alpha_keys = [
            "lora_unet-single_transformer_blocks-0-attn-to_q.alpha",
            "lora_unet_single_transformer_blocks_0_attn_to_q.alpha",
        ]
        self.rank_idx0_keys = [
            "lora_unet-single_transformer_blocks-0-attn-to_q.lora_down.weight",
            "lora_unet_single_transformer_blocks_0_attn_to_q.lora_down.weight",
        ]

    def __call__(
        self,
        lora_path: Union[str, Path, None] = None,
        state_dict: Optional[dict] = None,
        save_path: Union[str, Path, None] = None,
        up_multiplier: Optional[float] = 1.0,
    ):
        if state_dict is None and lora_path is not None:
            state_dict = load_file(lora_path)

        if up_multiplier is None:
            up_multiplier = self.get_multiplier(state_dict)

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".alpha"):
                continue

            orig_dtype = value.dtype

            new_val = value.float() * up_multiplier
            new_key = key
            new_key = new_key.replace("-", "_")
            new_key = new_key.replace("lora_unet_", "transformer.")
            for i in range(100):
                new_key = new_key.replace(
                    f"transformer_blocks_{i}_", f"transformer_blocks.{i}."
                )
            new_key = new_key.replace("lora_down", "lora_A")
            new_key = new_key.replace("lora_up", "lora_B")
            new_key = new_key.replace("_lora", ".lora")
            new_key = new_key.replace("attn_", "attn.")
            new_key = new_key.replace("ff_", "ff.")
            new_key = new_key.replace("context_net_", "context.net.")
            new_key = new_key.replace("0_proj", "0.proj")
            new_key = new_key.replace("norm_linear", "norm.linear")
            new_key = new_key.replace("norm_out_linear", "norm_out.linear")
            new_key = new_key.replace("to_out_", "to_out.")

            new_state_dict[new_key] = new_val.to(orig_dtype)

        if save_path is None:
            save_path = Path(lora_path).parent / f"{Path(lora_path).stem}.safetensors"
        self.save_model(save_path, new_state_dict)
        return save_path

    def get_multiplier(self, state_dict):
        alpha, rank = None, None
        for key in self.rank_idx0_keys:
            if key in state_dict:
                rank = int(state_dict[key].shape[0])
                break

        if rank is None:
            raise ValueError("Could not find rank in state dict")

        for key in self.alpha_keys:
            if key in state_dict:
                alpha = int(state_dict[key])
                break

        if alpha is None:
            alpha = rank

        return alpha / rank

    def save_model(self, save_path: Union[str, Path], state_dict):
        save_file(state_dict, save_path)
        print(f"Convert successful! Saved to: {save_path}")


if __name__ == "__main__":
    lora_path = "outputs/flux-hair_sliders/weights/flux-hair_sliders_latest.safetensors"
    convert = ConvertLoRAToPEFT()
    convert(lora_path=lora_path, up_multiplier=0.125)
