import torch
import os
import logging
import collections
import folder_paths
from transformers import AutoProcessor, SiglipVisionModel
from PIL import Image
import numpy as np
from .attention_processor_advanced import IPAFluxAttnProcessor2_0Advanced
from .utils import is_model_patched, FluxUpdateModules

import latent_preview
import comfy.samplers
import comfy.sample
from .ops import GGMLOps, move_patch_to_device
from .loader import gguf_sd_loader, gguf_clip_loader
from .dequant import is_quantized, is_torch_compatible

MODELS_DIR = os.path.join(folder_paths.models_dir, "ipadapter-flux")
if "ipadapter-flux" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter-flux"]
folder_paths.folder_names_and_paths["ipadapter-flux"] = (current_paths, folder_paths.supported_pt_extensions)

class GGUFModelPatcher(comfy.model_patcher.ModelPatcher):
    patch_on_device = False

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return
        weight = comfy.utils.get_attr(self.model, key)

        try:
            from comfy.lora import calculate_weight
        except Exception:
            calculate_weight = self.calculate_weight

        patches = self.patches[key]
        if is_quantized(weight):
            out_weight = weight.to(device_to)
            patches = move_patch_to_device(patches, self.load_device if self.patch_on_device else self.offload_device)
            # TODO: do we ever have legitimate duplicate patches? (i.e. patch on top of patched weight)
            out_weight.patches = [(calculate_weight, patches, key)]
        else:
            inplace_update = self.weight_inplace_update or inplace_update
            if key not in self.backup:
                self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(
                    weight.to(device=self.offload_device, copy=inplace_update), inplace_update
                )

            if device_to is not None:
                temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
            else:
                temp_weight = weight.to(torch.float32, copy=True)

            out_weight = calculate_weight(patches, temp_weight, key)
            out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        if unpatch_weights:
            for p in self.model.parameters():
                if is_torch_compatible(p):
                    continue
                patches = getattr(p, "patches", [])
                if len(patches) > 0:
                    p.patches = []
        # TODO: Find another way to not unload after patches
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    mmap_released = False
    def load(self, *args, force_patch_weights=False, **kwargs):
        # always call `patch_weight_to_device` even for lowvram
        super().load(*args, force_patch_weights=True, **kwargs)

        # make sure nothing stays linked to mmap after first load
        if not self.mmap_released:
            linked = []
            if kwargs.get("lowvram_model_memory", 0) > 0:
                for n, m in self.model.named_modules():
                    if hasattr(m, "weight"):
                        device = getattr(m.weight, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
                    if hasattr(m, "bias"):
                        device = getattr(m.bias, "device", None)
                        if device == self.offload_device:
                            linked.append((n, m))
                            continue
            if linked:
                print(f"Attempting to release mmap ({len(linked)})")
                for n, m in linked:
                    # TODO: possible to OOM, find better way to detach
                    m.to(self.load_device).to(self.offload_device)
            self.mmap_released = True

    def clone(self, *args, **kwargs):
        src_cls = self.__class__
        self.__class__ = GGUFModelPatcher
        n = super().clone(*args, **kwargs)
        n.__class__ = GGUFModelPatcher
        self.__class__ = src_cls
        # GGUF specific clone values below
        n.patch_on_device = getattr(self, "patch_on_device", False)
        return n


class MLPProjModelAdvanced(torch.nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class InstantXFluxIPAdapterModelAdvanced:
    def __init__(self, image_encoder_path, ip_ckpt, device, num_tokens=4):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = num_tokens
        # load image encoder
        self.image_encoder = SiglipVisionModel.from_pretrained(self.image_encoder_path).to(self.device, dtype=torch.float16)
        self.clip_image_processor = AutoProcessor.from_pretrained(self.image_encoder_path)
        # state_dict
        self.state_dict = torch.load(os.path.join(MODELS_DIR,self.ip_ckpt), map_location="cpu")
        self.joint_attention_dim = 4096
        self.hidden_size = 3072

    def init_proj(self):
        self.image_proj_model = MLPProjModelAdvanced(
            cross_attention_dim=self.joint_attention_dim, # 4096
            id_embeddings_dim=1152, 
            num_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)

    def set_ip_adapter_advanced(self, flux_model, weight_params, timestep_percent_range=(0.0, 1.0)):
        weight_start, weight_end, steps = weight_params
        s = flux_model.model_sampling
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        timestep_range = (percent_to_timestep_function(timestep_percent_range[0]), percent_to_timestep_function(timestep_percent_range[1]))
        ip_attn_procs = {}
        dsb_count = len(flux_model.diffusion_model.double_blocks)
        for i in range(dsb_count):
            name = f"double_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0Advanced(
                    hidden_size=self.hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale_start=weight_start,
                    scale_end=weight_end,
                    total_steps=steps,
                    timestep_range=timestep_range
                ).to(self.device, dtype=torch.float16)
        ssb_count = len(flux_model.diffusion_model.single_blocks)
        for i in range(ssb_count):
            name = f"single_blocks.{i}"
            ip_attn_procs[name] = IPAFluxAttnProcessor2_0Advanced(
                    hidden_size=self.hidden_size,
                    cross_attention_dim=self.joint_attention_dim,
                    num_tokens=self.num_tokens,
                    scale_start=weight_start,
                    scale_end=weight_end,
                    total_steps=steps,
                    timestep_range=timestep_range
                ).to(self.device, dtype=torch.float16)
        return ip_attn_procs
    
    def load_ip_adapter_advanced(self, flux_model, weight, timestep_percent_range=(0.0, 1.0)):
        self.image_proj_model.load_state_dict(self.state_dict["image_proj"], strict=True)
        ip_attn_procs = self.set_ip_adapter_advanced(flux_model, weight, timestep_percent_range)
        ip_layers = torch.nn.ModuleList(ip_attn_procs.values())
        ip_layers.load_state_dict(self.state_dict["ip_adapter"], strict=True)
        return ip_attn_procs

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=self.image_encoder.dtype)).pooler_output
            clip_image_embeds = clip_image_embeds.to(dtype=torch.float16)
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)

        # 回收显存
        # clip_image.to('cpu')
        # del clip_image
        # clip_image_embeds.to('cpu')
        # del clip_image_embeds
        # self.image_encoder.to('cpu')
        # del self.image_encoder
        # import gc
        # gc.collect()
        # torch.cuda.empty_cache()

        return image_prompt_embeds

class IPAdapterFluxLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "ipadapter": (folder_paths.get_filename_list("ipadapter-flux"),),
                "provider": (["cuda", "cpu", "mps"],),
            }
        }
    RETURN_TYPES = ("IP_ADAPTER_FLUX_INSTANTX",)
    RETURN_NAMES = ("ipadapterFlux",)
    FUNCTION = "load_model_advanced"
    CATEGORY = "InstantXNodes"

    def load_model_advanced(self, ipadapter, provider):
        logging.info("Loading InstantX IPAdapter Flux model.")
        clip_path = os.path.join(folder_paths.models_dir, "clip", "siglip-so400m-patch14-384")
        model = InstantXFluxIPAdapterModelAdvanced(image_encoder_path=clip_path, ip_ckpt=ipadapter, device=provider, num_tokens=128)
        return (model,)

class ApplyIPAdapterFluxAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter_flux": ("IP_ADAPTER_FLUX_INSTANTX", ),
                "image": ("IMAGE", ),
                "weight_start": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "weight_end": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05 }),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter_flux_advanced"
    CATEGORY = "InstantXNodes"

    def apply_ipadapter_flux_advanced(self, model, ipadapter_flux, image, weight_start, weight_end, steps, start_percent, end_percent):
        # Clean up old processors if they exist
        if hasattr(model.model, '_ip_attn_procs'):
            for proc in model.model._ip_attn_procs.values():
                proc.clear_memory()  # Add a new method for cleanup
            del model.model._ip_attn_procs

        pil_image = image.numpy()[0] * 255.0
        pil_image = Image.fromarray(pil_image.astype(np.uint8))
        ipadapter_flux.init_proj()
        
        IPAFluxAttnProcessor2_0Advanced.reset_all_instances()
        
        ip_attn_procs = ipadapter_flux.load_ip_adapter_advanced(model.model, (weight_start, weight_end, steps), (start_percent, end_percent))
        
        image_prompt_embeds = ipadapter_flux.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=None
        )
        is_patched = is_model_patched(model.model)
        bi = model.clone()
        FluxUpdateModules(bi, ip_attn_procs, image_prompt_embeds, is_patched)
        
        # Store reference to processors for cleanup
        bi.model._ip_attn_procs = ip_attn_procs

        return (bi,)


class SamplerCustomAdvancedPlus:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL", ),
                    "noise": ("NOISE", ),
                    "guider": ("GUIDER", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, noise, guider, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out

        # 清理IPA字典，避免显存泄露
        for k,v in model.model._ip_attn_procs.items():
            v.to('cpu')
        del model.model._ip_attn_procs
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return (out, out_denoised)


class SeedPlus:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }
        }

    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("seed", )

    FUNCTION = "get_noise"

    def get_noise(self, noise_seed):
        seed = np.random.randint(0, 2 ** 32 - 1)
        return (seed,)


class UnetLoaderGGUFPlus:
    @classmethod
    def INPUT_TYPES(s):
        unet_names = [x for x in folder_paths.get_filename_list("unet_gguf")]
        return {
            "required": {
                "seed": ("INT",),
                "unet_name": (unet_names,),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "bootleg"
    TITLE = "Unet Loader (GGUF)"

    def load_unet(self, seed, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None):
        ops = GGMLOps()

        if dequant_dtype in ("default", None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype in ["target"]:
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

        if patch_dtype in ("default", None):
            ops.Linear.patch_dtype = None
        elif patch_dtype in ["target"]:
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)

        # init model
        unet_path = folder_paths.get_full_path("unet", unet_name)
        sd = gguf_sd_loader(unet_path)
        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}
        )
        if model is None:
            logging.error("ERROR UNSUPPORTED UNET {}".format(unet_path))
            raise RuntimeError("ERROR: Could not detect model type of: {}".format(unet_path))
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device
        return (model,)


class IPAdapterFluxLoaderAdvancedPlus:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "seed": ("INT", ),
                "ipadapter": (folder_paths.get_filename_list("ipadapter-flux"),),
                "provider": (["cuda", "cpu", "mps"],),
            }
        }
    RETURN_TYPES = ("IP_ADAPTER_FLUX_INSTANTX",)
    RETURN_NAMES = ("ipadapterFlux",)
    FUNCTION = "load_model_advanced"
    CATEGORY = "InstantXNodes"

    def load_model_advanced(self, seed, ipadapter, provider):
        logging.info("Loading InstantX IPAdapter Flux model.")
        clip_path = os.path.join(folder_paths.models_dir, "clip", "siglip-so400m-patch14-384")
        model = InstantXFluxIPAdapterModelAdvanced(image_encoder_path=clip_path, ip_ckpt=ipadapter, device=provider, num_tokens=128)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "IPAdapterFluxLoaderAdvanced": IPAdapterFluxLoaderAdvanced,
    "ApplyIPAdapterFluxAdvanced": ApplyIPAdapterFluxAdvanced,
    "SamplerCustomAdvancedPlus": SamplerCustomAdvancedPlus,
    "SeedPlus": SeedPlus,
    "UnetLoaderGGUFPlus": UnetLoaderGGUFPlus,
    "IPAdapterFluxLoaderAdvancedPlus": IPAdapterFluxLoaderAdvancedPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterFluxLoaderAdvanced": "Load IPAdapter Flux Model (Advanced)",
    "ApplyIPAdapterFluxAdvanced": "Apply IPAdapter Flux Model (Advanced)",
    "SamplerCustomAdvancedPlus": "Sampler Custom Advanced (Plus)",
    "SeedPlus": "Seed Plus",
    "UnetLoaderGGUFPlus": "Unet Loader GGUF Plus",
    "IPAdapterFluxLoaderAdvancedPlus": "Load IPAdapter Flux Model (Plus)"
}

