import torch
from PIL import Image
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from pathlib import Path


class BackgroundRemover:
    """Remove backgrounds from images using BiRefNet segmentation model."""
    
    def __init__(self, device=None, model_name='ZhengPeng7/BiRefNet', use_half_precision=True):
        """
        Initialize the background remover.
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
            model_name: Hugging Face model identifier
            use_half_precision: Use half precision (FP16) for faster inference
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_half_precision = use_half_precision and self.device == 'cuda'
        
        print(f"Loading BiRefNet model on {self.device}...")
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        if self.use_half_precision:
            torch.set_float32_matmul_precision(['high', 'highest'][0])
            self.model.half()
        
        print("Model loaded successfully!")
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def remove_background(self, image_path, output_path=None, threshold=0.5):
        """
        Remove background from an image.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save the output image
            threshold: Confidence threshold for mask (0-1)
        
        Returns:
            tuple: (PIL Image with transparent background, mask)
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Preprocess
        input_tensor = self.transform(image).unsqueeze(0)
        if self.use_half_precision:
            input_tensor = input_tensor.half()
        input_tensor = input_tensor.to(self.device)
        
        # Predict
        print(f"Processing {Path(image_path).name}...")
        with torch.no_grad():
            pred = self.model(input_tensor)[-1].sigmoid()
        
        # Post-process
        mask_tensor = pred.cpu().squeeze()
        
        # Apply threshold
        mask_tensor = (mask_tensor > threshold).float()
        
        # Convert to PIL and resize to original dimensions
        mask_pil = transforms.ToPILImage()(mask_tensor)
        mask_pil = mask_pil.resize(original_size, Image.LANCZOS)
        
        # Apply mask to original image
        image.putalpha(mask_pil)
        
        # Save if output path provided
        if output_path:
            image.save(output_path)
            print(f"Saved to {output_path}")
        
        return image, mask_pil
    
    def batch_remove_background(self, input_dir, output_dir, threshold=0.5):
        """
        Remove backgrounds from all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images
            threshold: Confidence threshold for mask (0-1)
        
        Returns:
            list: Paths to generated images
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return []
        
        output_files = []
        for i, img_file in enumerate(image_files, 1):
            try:
                output_file = output_path / f"{img_file.stem}_transparent.png"
                self.remove_background(str(img_file), str(output_file), threshold)
                output_files.append(str(output_file))
                print(f"✓ Processed {i}/{len(image_files)}")
            except Exception as e:
                print(f"✗ Error processing {img_file.name}: {e}")
        
        print(f"\nCompleted! Processed {len(output_files)} images.")
        return output_files


# Example usage:
if __name__ == "__main__":
    # Single image
    remover = BackgroundRemover()
    image_with_transparency, mask = remover.remove_background(
        image_path='/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/160_poses_grab/dollhouse_project/visuals/characters/femalesprite.png',
        output_path='/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/160_poses_grab/dollhouse_project/visuals/characters/spritesheet_variants/femalesprite_transparent.png'
    )