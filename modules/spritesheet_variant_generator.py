import os
import torch
from rich import print as rp
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from controlnet_aux import OpenposeDetector

from Real_ESRGAN.RealESRGAN import RealESRGAN
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
try:
    from modules.shimsalabim import ShimSalaBim
except:
    from shimsalabim import ShimSalaBim

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
if not torch:
    import torch



class SpriteSheetProcessor:
    """Process sprite sheets: extract poses and generate new variants"""
    
    def __init__(
        self,
        checkpoint_pose: str,
        safe_pose_path: str,
        model_checkpoint: str,
        upscaler_path: str,
        torch_dtype=torch.float16,
        device: str = None,
        upscale_factor: int = 2
    ):
        """
        Initialize the sprite sheet processor
        
        Args:
            checkpoint_pose: Path to ControlNet pose checkpoint
            model_checkpoint: Path to base diffusion model checkpoint
            upscaler_path: Path to RealESRGAN upscaler weights
            torch_dtype: PyTorch dtype for models
            device: Device to use ('cuda' or 'cpu')
            upscale_factor: Upscaling factor (2, 4, or 8)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_dtype = torch_dtype
        self.checkpoint_pose = checkpoint_pose
        self.model_checkpoint = model_checkpoint
        self.upscaler_path = upscaler_path
        self.upscale_factor = upscale_factor
        self.save_path = safe_pose_path
        # Initialize upscaler
        self.upscaler = RealESRGAN(self.device, scale=2)
        self.upscaler.load_weights(upscaler_path, download=False)
        
        # Initialize pose processor
        self.pose_processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        
        # Initialize ControlNet and pipeline
        self._init_pipeline()
        
        rp(f"[green]✓ Processor initialized on {self.device}[/green]")
    
    def _init_pipeline(self):
        """Initialize the diffusion pipeline with ControlNet"""
        controlnet = ControlNetModel.from_single_file(
            self.checkpoint_pose,
            safety_checker=None,
            feature_extractor=None,
            torch_dtype=self.torch_dtype
        )
        
        self.pipe = StableDiffusionControlNetPipeline.from_single_file(
            self.model_checkpoint,
            controlnet=controlnet,
            safety_checker=None,
            feature_extractor=None,
            torch_dtype=self.torch_dtype
        )
        
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Load LoRA if available
        try:
            self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
            self.pipe.fuse_lora()
        except Exception as e:
            rp(f"[yellow]LoRA loading warning: {e}[/yellow]")
        
        # Optimization settings
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.to(self.device)
        self.pipe.enable_xformers_memory_efficient_attention()
    
    def upscale_image(self, image: Image.Image) -> Image.Image:
        """Upscale image using RealESRGAN"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            result = self.upscaler.predict(image.convert('RGB'))
        except torch.cuda.OutOfMemoryError as e:
            rp(f"[red]OOM Error during upscaling: {e}[/red]")
            self.upscaler = RealESRGAN(self.device, scale=2)
            self.upscaler.load_weights(self.upscaler_path, download=False)
            result = self.upscaler.predict(image.convert('RGB'))
        
        return result
    
    def extract_poses(self, image: Image.Image, save_path: str = None) -> Image.Image:
        """
        Extract poses from sprite sheet using OpenPose
        
        Args:
            image: Input sprite sheet image
            save_path: Optional path to save the pose extraction
            
        Returns:
            Pose control image
        """
        original_w, original_h = image.size
        rp(f"[cyan]Original size: {original_w} x {original_h}[/cyan]")
        
        # Upscale for better pose detection
        upscaled = self.upscale_image(image)
        rp(f"[cyan]Upscaled size: {upscaled.size}[/cyan]")
        
        # Extract poses
        control_image = self.pose_processor(upscaled, hand_and_face=True)
        
        # Scale back to original size
        control_image = control_image.resize(
            (original_w, original_h),
            Image.Resampling.LANCZOS
        )
        
        
        # Save if path provided
        if self.save_path:
            control_image.save(self.save_path)
            rp(f"[green]✓ Pose extraction saved: {save_path}[/green]")
        
        return control_image
    
    def generate_variants(
        self,
        prompt: str,
        control_image: Image.Image,
        original_size: tuple,
        num_variants: int = 10,
        num_inference_steps: int = 32,
        guidance_scale: float = 4.5,
        strength: float = 0.75,
        seed: int = 666875646756233,
        output_dir: str = "./"
    ) -> list:
        """
        Generate variant sprite sheets using pose control
        
        Args:
            prompt: Text prompt for generation
            control_image: Pose control image
            original_size: Tuple of (width, height) for output
            num_variants: Number of variants to generate
            num_inference_steps: Diffusion steps
            guidance_scale: Classifier-free guidance scale
            strength: ControlNet conditioning strength
            seed: Base seed for generation
            output_dir: Directory to save outputs
            
        Returns:
            List of generated images
        """
        os.makedirs(output_dir, exist_ok=True)
        
        negative_prompt = (
            "shadows, floor, background, cartoon, low quality, bad anatomy, bad hands, text, error, "
            "missing fingers, extra digit, fewer digits, cropped, worst quality, "
            "low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
        )
        
        width, height = original_size
        generated_images = []
        
        rp(f"[cyan]Generating {num_variants} variants with prompt: '{prompt}'[/cyan]")
        
        for i in range(num_variants):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            generator = torch.manual_seed(seed + i)
            variant_prompt = f"{prompt}, variant {i+1}"
            
            try:
                image = self.pipe(
                    variant_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=strength,
                    generator=generator,
                    image=control_image,
                    height=height,
                    width=width
                ).images[0]
                
                output_path = os.path.join(output_dir, f"variant_{i:02d}.png")
                image.save(output_path)
                generated_images.append(image)
                
                rp(f"[green]✓ Generated variant {i+1}/{num_variants}[/green]")
                
            except RuntimeError as e:
                rp(f"[red]Error generating variant {i+1}: {e}[/red]")
        
        return generated_images
    
    def process_sprite_sheet(
        self,
        input_image_path: str,
        prompt: str,
        num_variants: int = 10,
        output_pose_path: str = None,
        output_dir: str = "./output",
        seed: int = 666875646756233,
        **generation_kwargs
    ) -> dict:
        """
        Complete pipeline: load image, extract poses, generate variants
        
        Args:
            input_image_path: Path to input sprite sheet
            prompt: Text prompt for generation
            num_variants: Number of variants to generate
            output_pose_path: Path to save pose extraction
            output_dir: Directory for variant outputs
            seed: Base seed for generation
            **generation_kwargs: Additional arguments for generate_variants
            
        Returns:
            Dictionary with results
        """
        rp(f"[bold cyan]Processing sprite sheet: {input_image_path}[/bold cyan]")
        
        # Load image
        image = load_image(input_image_path)
        width, height = image.size
        
        # Extract poses
        if output_pose_path is None:
            base_name = Path(input_image_path).stem
            output_pose_path = f"{base_name}_poses.png"
        
        control_image = self.extract_poses(image, save_path=output_pose_path)
        
        # Generate variants
        variants = self.generate_variants(
            prompt=prompt,
            control_image=control_image,
            original_size=(width, height),
            num_variants=num_variants,
            seed=seed,
            output_dir=output_dir,
            **generation_kwargs
        )
        
        return {
            "input_path": input_image_path,
            "pose_extraction_path": output_pose_path,
            "variants_dir": output_dir,
            "num_variants_generated": len(variants),
            "prompt": prompt,
            "original_size": (width, height)
        }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":





    # Configuration

    # t2i models
    xenogasm_model="/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/models/sd15_models/uberRealisticPornMerge_v23Final.safetensors"
    dreamshaper8="/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/models/sd15_models/dreamshaper_8.safetensors"


    root_folder = "/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend"
    #remote_checkpoint = "lllyasviel/control_v11p_sd15_openpose"

    UPSCALE_FACTOR = 2
    CHECKPOINT_POSE =  f"{root_folder}/models/controlnets/control_v11p_sd15_openpose_fp16.safetensors"
    UPSCALER_PATH = '/media/codemonkeyxl/DATA2/new_comfyui/ComfyUI/models/upscale_models/RealESRGAN_x2plus.pth'
    INPUT_IMAGE = '/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/160_poses_grab/dollhouse_project/visuals/characters/Screenshot from 2026-01-09 08-45-11.png'
    PROMPT = "a walking female, naked, huge breasts, short hair"
    SEED = 666875646756233

    spite_variant_prompt=f"a 2D character spritesheet with of {PROMPT}"
    variant_negative_prompt = "shadows, floor, background, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    MODEL_CHECKPOINT = xenogasm_model


    # Initialize processor
    processor = SpriteSheetProcessor(
        checkpoint_pose=CHECKPOINT_POSE,
        model_checkpoint=MODEL_CHECKPOINT,
        safe_pose_path=INPUT_IMAGE.replace('.png', '_pose.png'),
        upscaler_path=UPSCALER_PATH,
        upscale_factor=UPSCALE_FACTOR
    )
    
    # Process sprite sheet
    results = processor.process_sprite_sheet(
        input_image_path=INPUT_IMAGE,
        prompt=spite_variant_prompt,
        num_variants=1,
        output_pose_path=INPUT_IMAGE.replace('.png', '_pose.png'),
        output_dir="./sprite_variants",
        seed=SEED,
        num_inference_steps=32,
        guidance_scale=5.5,
        strength=0.95
    )
    
    rp(f"\n[bold green]Processing complete![/bold green]")
    rp(f"Pose extraction: {results['pose_extraction_path']}")
    rp(f"Variants saved to: {results['variants_dir']}")
    rp(f"Total variants generated: {results['num_variants_generated']}")