"""
PipelineHarvester: Minimal, proven pipeline loading via ShimSalaBim.
No scheduler morphing complexity. Single pipe, reuse components.
"""

from pathlib import Path
from typing import Optional, Union
from PIL import Image

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
torch = shim.torch.nn
torchvision = shim.torchvision
langchain_huggingface = shim.langchain_huggingface
langchain_community = shim.langchain_community
accelerate = shim.accelerate
safetensors = shim.safetensors
gguf = shim.gguf

if not Llama:
    from llama_cpp import Llama



class PipelineHarvester:
    """Minimal pipeline harvester. Tested working."""
    
    def __init__(
        self,
        model_path: str = "/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/models/sd15_models/dreamshaper_8.safetensors",
        device: str = "cuda",
    ):
        self.model_path = model_path
        self.device = device
        
        # Load ONCE as inpaint (has all components)
        print(f"Loading {Path(model_path).name}...")
        
        from diffusers import StableDiffusionPipeline
        
        self.pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            feature_extractor=None,
        )
        
        # Optimize
        self.pipe.enable_vae_slicing()
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✓ xformers enabled")
        except:
            print("⚠ xformers skipped")
        
        self.pipe.to(device)
        print(f"✓ Loaded on {device}")
    
    def generate_inpaint(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image,
        strength: float = 0.8,
        num_inference_steps: int = 4,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Inpaint with mask"""
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(mask)
        
        if mask.mode != "L":
            mask = mask.convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.LANCZOS)
        
        generator = torch.manual_seed(seed) if seed else None
        
        print(f"inpaint: {prompt[:40]}")
        
        with torch.no_grad():
            output = self.pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                generator=generator,
            )
        
        return output.images[0]
    
    def generate_i2i(
        self,
        prompt: str,
        image: Image.Image,
        strength: float = 0.6,
        num_inference_steps: int = 4,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Image-to-image via inpaint (full white mask)"""
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Full white mask = i2i
        mask = Image.new("L", image.size, 255)
        
        generator = torch.manual_seed(seed) if seed else None
        
        print(f"i2i: {prompt[:40]}")
        
        with torch.no_grad():
            output = self.pipe(
                prompt=prompt,
                image=image,
                mask_image=mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                generator=generator,
            )
        
        return output.images[0]
    
    def load_image_from_path(self, path: Union[str, Path]) -> Image.Image:
        """Load image"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path).convert("RGB")
    
    def cleanup(self):
        """Free memory"""
        if self.pipe:
            del self.pipe
        torch.cuda.empty_cache()
        print("✓ Cleanup")


if __name__ == "__main__":
    h = PipelineHarvester(device="cuda")
    
    # Test i2i
    print("\n=== Test i2i ===")
    img = Image.new("RGB", (512, 512), (200, 200, 200))
    result = h.generate_i2i("add a blue sofa", img, seed=42)
    result.save("test.png")
    print("✓ Done")
    
    h.cleanup()