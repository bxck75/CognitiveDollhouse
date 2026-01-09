import cv2
import numpy as np
from scipy.ndimage import convolve
from pathlib import Path


class CannyEdgeDetector:
    """Compact Canny edge detection with automatic saving"""
    
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        """
        Initialize Canny edge detector
        
        Args:
            kernel_size: Gaussian kernel size (odd number, default 3)
            sigma: Gaussian sigma value (default 1.0)
        """
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma
    
    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        """Generate Gaussian kernel"""
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        return np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    
    @staticmethod
    def _sobel_filters(img: np.ndarray) -> tuple:
        """Apply Sobel filters"""
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        
        Ix = convolve(img, Kx)
        Iy = convolve(img, Ky)
        
        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        
        return G, theta
    
    @staticmethod
    def _non_max_suppression(img: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Non-maximum suppression"""
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = theta * 180.0 / np.pi
        angle[angle < 0] += 180
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                try:
                    q = r = 255
                    
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q, r = img[i, j+1], img[i, j-1]
                    elif (22.5 <= angle[i,j] < 67.5):
                        q, r = img[i+1, j-1], img[i-1, j+1]
                    elif (67.5 <= angle[i,j] < 112.5):
                        q, r = img[i+1, j], img[i-1, j]
                    elif (112.5 <= angle[i,j] < 157.5):
                        q, r = img[i-1, j-1], img[i+1, j+1]
                    
                    Z[i,j] = img[i,j] if (img[i,j] >= q) and (img[i,j] >= r) else 0
                except IndexError:
                    pass
        
        return Z
    
    @staticmethod
    def _threshold(img: np.ndarray, low_ratio: float = 0.05, high_ratio: float = 0.09) -> np.ndarray:
        """Double threshold"""
        high_threshold = img.max() * high_ratio
        low_threshold = high_threshold * low_ratio
        
        M, N = img.shape
        res = np.zeros((M, N), dtype=np.int32)
        
        strong_i, strong_j = np.where(img >= high_threshold)
        weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))
        
        res[strong_i, strong_j] = 255
        res[weak_i, weak_j] = 25
        
        return res
    
    @staticmethod
    def _hysteresis(img: np.ndarray) -> np.ndarray:
        """Edge tracking by hysteresis"""
        M, N = img.shape
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                if img[i,j] == 25:
                    if np.any(img[i-1:i+2, j-1:j+2] == 255):
                        img[i,j] = 255
                    else:
                        img[i,j] = 0
        
        return img
    
    def detect(self, img: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection"""
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gaussian = convolve(img_gray, self._gaussian_kernel(self.kernel_size, self.sigma))
        G, theta = self._sobel_filters(img_gaussian)
        img_nonmax = self._non_max_suppression(G, theta)
        img_threshold = self._threshold(img_nonmax)
        img_final = self._hysteresis(img_threshold)
        
        return img_final
    
    def process_and_save(
        self,
        input_path: str,
        output_path: str = None
    ) -> str:
        """
        Process image and save result
        
        Args:
            input_path: Path to input image
            output_path: Path to save output (default: input_filename_canny.png)
            
        Returns:
            Path to saved output image
        """
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {input_path}")
        
        edges = self.detect(img)
        
        if output_path is None:
            input_file = Path(input_path)
            output_path = str(input_file.parent / f"{input_file.stem}_canny.png")
        
        cv2.imwrite(output_path, edges)
        print(f"✓ Canny edge detection saved to: {output_path}")
        
        return output_path


# Usage Example
if __name__ == "__main__":
    detector = CannyEdgeDetector(kernel_size=3, sigma=5.4)
    
    # Process image
    output = detector.process_and_save(
        input_path="/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/160_poses_grab/dollhouse_project/visuals/characters/ladysprite.png",
        output_path="/media/codemonkeyxl/B500/coding_folder/visual_chatbot/backend/160_poses_grab/dollhouse_project/visuals/characters/ladysprite_canny.png"
    )
    
    # Or let it auto-generate output path
    #detector.process_and_save("spritesheet.png")