"""
PipelineHarvester v3: Fast pipeline morphing with aggressive memory management.
Key: Clean intermediates after each generation, empty cache frequently.
"""

import os
import gc
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)

from modules.shimsalabim import ShimSalaBim
    
global_packages_folder = '/home/codemonkeyxl/.local/lib/python3.10/site-packages'

global_pkgs = [
    ('llama_cpp', global_packages_folder),
    ('torch', global_packages_folder),
    ('torchvision', global_packages_folder),
    ('langchain', global_packages_folder),
    ('langchain_community', global_packages_folder),
    ('accelerate', global_packages_folder),
    ('safetensors', global_packages_folder),
    ('gguf', global_packages_folder),
]

shim = ShimSalaBim(global_pkgs, classes_to_wrap={})

Llama = shim.llama_cpp.Llama
torch = shim.torch

if not Llama:
    from llama_cpp import Llama

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    DPMSolverMultistepScheduler,
)

# ============================================================================
# MEMORY UTILITIES
# ============================================================================

def clear_memory():
    """Aggressively clear GPU/CPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# ============================================================================
# REALESRGAN UPSCALER
# ============================================================================

class RealESRGANUpscaler:
    """2x upscaler using RealESRGAN weights"""
    
    def __init__(
        self,
        weights_path: str = "/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/weights/RealESRGAN_x2.pth",
        device: str = "cuda",
    ):
        self.device = device
        self.weights_path = Path(weights_path)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load RealESRGAN model"""
        if not self.weights_path.exists():
            logger.warning(f"⚠ RealESRGAN weights not found: {self.weights_path}")
            return
        
        try:
            try:
                from realesrgan import RRDBNet
            except ImportError:
                logger.warning("⚠ basicsr/realesrgan not installed, using fallback bicubic upscale")
                self.model = None
                return
            
            self.model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            
            if "params" in checkpoint:
                self.model.load_state_dict(checkpoint["params"])
            elif "params_ema" in checkpoint:
                self.model.load_state_dict(checkpoint["params_ema"])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("✓ RealESRGAN x2 loaded")
        
        except Exception as e:
            logger.warning(f"⚠ Failed to load RealESRGAN: {e}")
            self.model = None
    
    @torch.no_grad()
    def upscale(self, image: Image.Image) -> Image.Image:
        """Upscale image 2x"""
        if self.model is None:
            return image
        
        try:
            img_np = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            output = self.model(img_tensor)
            
            output_np = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            output_pil = Image.fromarray(output_np, "RGB")
            
            # Clean up intermediate tensors
            del img_tensor, output
            clear_memory()
            
            return output_pil
        except Exception as e:
            logger.warning(f"⚠ Upscale failed: {e}")
            return image


# ============================================================================
# PIPELINE HARVESTER V3 - AGGRESSIVE GC
# ============================================================================

class PipelineHarvester:
    """
    Fast pipeline morphing for t2i, i2i, inpaint with aggressive memory cleanup.
    Clears intermediates after every generation.
    """
    
    def __init__(
        self,
        model_path: str = "/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/models/sd15_models/dreamshaper_8.safetensors",
        device: str = "cuda",
        dtype = torch.float16,
        enable_attention_slicing: bool = True,
        upscaler_weights: Optional[str] = None,
        enable_upscaling: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.enable_attention_slicing = enable_attention_slicing
        self.enable_upscaling = enable_upscaling
        
        self.current_pipe = None
        self.upscaler = None
        
        if enable_upscaling:
            if upscaler_weights is None:
                upscaler_weights = "/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/weights/RealESRGAN_x2.pth"
            self.upscaler = RealESRGANUpscaler(upscaler_weights, device)
        
        self._load_base_pipeline()
        logger.info(f"✓ PipelineHarvester ready on {device}")
    
    def _load_base_pipeline(self):
        """Load inpainting pipeline as base (has all components)"""
        logger.info(f"Loading from {self.model_path}...")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            self.current_pipe = StableDiffusionInpaintPipeline.from_single_file(
                self.model_path,
                torch_dtype=self.dtype,
                safety_checker=None,
                feature_extractor=None,
            )
            
            self.current_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.current_pipe.scheduler.config
            )
            
            self.current_pipe = self.current_pipe.to(self.device)
            
            if self.enable_attention_slicing:
                self.current_pipe.vae.enable_slicing()
                self.current_pipe.vae.enable_tiling()
                logger.info("✓ Slicing & tiling enabled")
            
            try:
                self.current_pipe.enable_xformers_memory_efficient_attention()
                logger.info("✓ xformers enabled")
            except Exception as e:
                logger.info(f"⚠ xformers unavailable: {e}")
            
            logger.info("✓ Pipeline loaded and optimized")
            clear_memory()
        
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise
    
    def _morph_to_t2i(self):
        """Morph current pipe to text-to-image"""
        logger.info("→ Morphing to t2i")
        self.current_pipe = StableDiffusionPipeline.from_pipe(
            self.current_pipe,
            safety_checker=None,
            feature_extractor=None,
        )
    
    def _morph_to_i2i(self):
        """Morph current pipe to image-to-image"""
        logger.info("→ Morphing to i2i")
        self.current_pipe = StableDiffusionImg2ImgPipeline.from_pipe(
            self.current_pipe,
            safety_checker=None,
            feature_extractor=None,
        )
    
    def _morph_to_inpaint(self):
        """Morph current pipe to inpainting"""
        logger.info("→ Morphing to inpaint")
        self.current_pipe = StableDiffusionInpaintPipeline.from_pipe(
            self.current_pipe,
            safety_checker=None,
            feature_extractor=None,
        )
    
    def generate_t2i(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 16,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Text-to-image generation"""
        self._morph_to_t2i()
        
        generator = torch.manual_seed(seed) if seed else None
        logger.info(f"t2i: {prompt}")
        
        with torch.no_grad():
            output = self.current_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        
        result = output.images[0]
        
        # Clean up output object and intermediates
        del output, generator
        clear_memory()
        
        return result
    
    def generate_i2i(
        self,
        prompt: str,
        image: Image.Image,
        negative_prompt: str = "",
        strength: float = 0.6,
        num_inference_steps: int = 16,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        upscale_first: bool = True,
    ) -> Image.Image:
        """Image-to-image modification (with optional upscaling)"""
        self._morph_to_i2i()
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        original_size = image.size
        if upscale_first and self.upscaler:
            logger.info(f"Upscaling {original_size} → 2x before i2i")
            image = self.upscaler.upscale(image)
        
        generator = torch.manual_seed(seed) if seed else None
        logger.info(f"i2i: {prompt} (strength={strength})")
        
        with torch.no_grad():
            output = self.current_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        
        result = output.images[0]
        
        # Downscale back to original if we upscaled
        if upscale_first and self.upscaler and result.size != original_size:
            logger.info(f"Downscaling back to {original_size}")
            result = result.resize(original_size, Image.LANCZOS)
        
        # Clean up
        del output, generator, image
        clear_memory()
        
        return result
    
    def generate_inpaint(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image,
        negative_prompt: str = "",
        strength: float = 0.8,
        num_inference_steps: int = 16,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        upscale_first: bool = True,
    ) -> Image.Image:
        """Inpainting with mask (with optional upscaling)"""
        self._morph_to_inpaint()
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(mask)
        
        original_size = image.size
        if upscale_first and self.upscaler:
            logger.info(f"Upscaling {original_size} → 2x before inpaint")
            image = self.upscaler.upscale(image)
            mask = mask.resize(image.size, Image.NEAREST)
        
        if mask.mode != "L":
            mask = mask.convert("L")
        
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.LANCZOS)
        
        generator = torch.manual_seed(seed) if seed else None
        logger.info(f"inpaint: {prompt} (strength={strength})")
        
        with torch.no_grad():
            output = self.current_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        
        result = output.images[0]
        
        # Downscale back to original if we upscaled
        if upscale_first and self.upscaler and result.size != original_size:
            logger.info(f"Downscaling back to {original_size}")
            result = result.resize(original_size, Image.LANCZOS)
        
        # Clean up
        del output, generator, image, mask
        clear_memory()
        
        return result
    
    def load_image_from_path(self, path: Union[str, Path]) -> Image.Image:
        """Load image from file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")
    
    def load_mask_from_path(self, path: Union[str, Path]) -> Image.Image:
        """Load mask from file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Mask not found: {path}")
        return Image.open(path).convert("L")
    
    def cleanup(self):
        """Free memory"""
        if self.current_pipe:
            del self.current_pipe
        if self.upscaler:
            if self.upscaler.model:
                del self.upscaler.model
        
        self.current_pipe = None
        self.upscaler = None
        
        clear_memory()
        logger.info("✓ Cleanup complete")