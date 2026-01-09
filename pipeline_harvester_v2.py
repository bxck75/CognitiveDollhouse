"""
PipelineHarvester: Fast pipeline morphing with LCM LoRA.
Loads from local safetensors, caches components, morphs between t2i/i2i/inpaint.
Works with ShimSalaBim for memory-efficient lib loading from root.
"""

import os
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# ShimSalaBim setup (your custom system)

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


# Specify which classes you want to monitor usage for
classes_to_monitor = {
    'llama_cpp': ['Llama'],
    'torch': ['nn.Module'],
    'torchvision': ['transforms.Compose'],
    'langchain_huggingface': ['transformers.pipeline'], 
    'langchain_community': ['embeddings.HuggingFaceEmbeddings'],
    'accelerate': ['AcceleratorState'],
    'safetensors': ['safetensors.torch.SFT', 'safetensors.torch.SFT.from_pretrained'],
    'gguf': ['gguf.GGUF'],
}

shim = ShimSalaBim(global_pkgs, classes_to_wrap={})

Llama = shim.llama_cpp.Llama
torch = shim.torch
torchvision = shim.torchvision
langchain_huggingface = shim.langchain_huggingface
langchain_community = shim.langchain_community
accelerate = shim.accelerate
safetensors = shim.safetensors
gguf = shim.gguf

if not Llama:
    from llama_cpp import Llama



# Now import diffusers classes
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    DPMSolverMultistepScheduler,
)


class PipelineHarvester:
    """
    Fast pipeline morphing for t2i, i2i, inpaint with LCM LoRA.
    Loads once, morphs between pipeline types, keeps single pipe in memory.
    Works with ShimSalaBim for memory-efficient library loading.
    """
    
    def __init__(
        self,
        model_path: str = "/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/models/sd15_models/dreamshaper_8.safetensors",
        device: str = "cuda",
        dtype = torch.float16,
        enable_attention_slicing: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.enable_attention_slicing = enable_attention_slicing
        
        self.current_pipe = None
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
            
            # Set scheduler
            self.current_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.current_pipe.scheduler.config
            )
            
            # Optimize
            if self.enable_attention_slicing:
                self.current_pipe.vae.enable_slicing()
                self.current_pipe.vae.enable_tiling()
                logger.info("✓ Slicing & tiling enabled")
            
            # Enable xformers if available
            try:
                self.current_pipe.enable_xformers_memory_efficient_attention()
                logger.info("✓ xformers enabled")
            except Exception as e:
                logger.info(f"⚠ xformers unavailable: {e}")
            
            # Move to device
            self.current_pipe = self.current_pipe.to(self.device)
            logger.info("✓ Pipeline loaded and optimized")
        
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
        ).to(self.device)
    
    def _morph_to_i2i(self):
        """Morph current pipe to image-to-image"""
        logger.info("→ Morphing to i2i")
        self.current_pipe = StableDiffusionImg2ImgPipeline.from_pipe(
            self.current_pipe,
            safety_checker=None,
            feature_extractor=None,
        ).to(self.device)
    
    def _morph_to_inpaint(self):
        """Morph current pipe to inpainting"""
        logger.info("→ Morphing to inpaint")
        self.current_pipe = StableDiffusionInpaintPipeline.from_pipe(
            self.current_pipe,
            safety_checker=None,
            feature_extractor=None,
        ).to(self.device)
    
    def generate_t2i(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 4,
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
        
        return output.images[0]
    
    def generate_i2i(
        self,
        prompt: str,
        image: Image.Image,
        negative_prompt: str = "",
        strength: float = 0.6,
        num_inference_steps: int = 4,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Image-to-image modification"""
        self._morph_to_i2i()
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
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
        
        return output.images[0]
    
    def generate_inpaint(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image,
        negative_prompt: str = "",
        strength: float = 0.8,
        num_inference_steps: int = 4,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Inpainting with mask"""
        self._morph_to_inpaint()
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(mask)
        
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
        
        return output.images[0]
    
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
        torch.cuda.empty_cache()
        logger.info("✓ Cleanup complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    harvester = PipelineHarvester(device="cuda", dtype=torch.float16)
    
    # Test t2i
    print("\n=== Test t2i ===")
    img = harvester.generate_t2i(
        prompt="minimalist living room with wooden floors",
        height=512, width=512, seed=42
    )
    img.save("test_t2i.png")
    print("✓ Saved test_t2i.png")
    
    # Test i2i
    print("\n=== Test i2i ===")
    img = harvester.generate_i2i(
        prompt="add plants and botanical theme",
        image=img, strength=0.9, seed=43
    )
    img.save("test_i2i.png")
    print("✓ Saved test_i2i.png")




    
    # Test inpaint
    print("\n=== Test inpaint ===")
    import numpy as np
    mask = Image.new("L", img.size, 0)
    mask_arr = np.array(mask)
    mask_arr[150:350, 150:350] = 255
    mask = Image.fromarray(mask_arr)
    
    img = harvester.generate_inpaint(
        prompt="vibrant blue sofa",
        image=img, mask=mask, strength=0.8, seed=44
    )
    img.save("test_inpaint.png")
    print("✓ Saved test_inpaint.png")




    
    harvester.cleanup()
    print("\n✓ Done!")