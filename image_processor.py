"""
Core Image Processing Module for Product Images
Handles background removal, shadow elimination, smart cropping, and resizing
"""

import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from rembg import remove, new_session
from typing import Tuple, Optional
import io
import torch
from torchvision import transforms
import requests
import base64


class ImageProcessor:
    """Main image processor for e-commerce product images"""

    # Class-level cache for BiRefNet model
    _birefnet_model = None
    _birefnet_transform = None

    def __init__(
        self,
        target_size: int = 1000,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        padding_percent: float = 5.0,
        crop_tightness: str = "normal",
        enable_shadow_removal: bool = True,
        enable_smart_crop: bool = True,
        save_intermediate: bool = False,
        intermediate_dir: str = "temp",
        bg_model: str = "u2net",
        alpha_matting: bool = False,
        post_process: bool = True,
        hf_api_token: Optional[str] = None,
    ):
        """
        Initialize the image processor

        Args:
            target_size: Target size for output images (square)
            background_color: RGB tuple for background color (default: white)
            padding_percent: Percentage of padding around product (0-15%)
            crop_tightness: Cropping mode - 'tight', 'normal', or 'loose'
            enable_shadow_removal: Enable shadow detection and removal (default: True)
            enable_smart_crop: Enable advanced contour-based cropping (default: True)
            save_intermediate: Save intermediate processing steps (default: False)
            intermediate_dir: Directory to save intermediate images (default: "temp")
            bg_model: Background removal model - 'u2net', 'isnet-general-use', 'u2netp', 'birefnet', etc.
            alpha_matting: Enable alpha matting for smoother edges (default: False)
            post_process: Post-process mask to remove artifacts (default: True)
            hf_api_token: HuggingFace API token for BiRefNet API (optional)
        """
        self.target_size = target_size
        self.background_color = background_color
        self.padding_percent = padding_percent / 100.0  # Convert to decimal
        self.crop_tightness = crop_tightness
        self.enable_shadow_removal = enable_shadow_removal
        self.enable_smart_crop = enable_smart_crop
        self.save_intermediate = save_intermediate
        self.intermediate_dir = intermediate_dir
        self.bg_model = bg_model
        self.alpha_matting = alpha_matting
        self.post_process = post_process
        self.hf_api_token = hf_api_token

        # Ensure intermediate directory exists if needed
        if self.save_intermediate:
            os.makedirs(self.intermediate_dir, exist_ok=True)

    def process_image(self, image_path: str, output_path: str) -> dict:
        """
        Complete processing pipeline for a product image

        Args:
            image_path: Path to input image
            output_path: Path to save processed image

        Returns:
            dict with processing stats and status
        """
        try:
            # Load image
            img = Image.open(image_path)
            original_size = img.size

            # Prepare base name for intermediate files
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            intermediate_paths = {}

            # Step 1: Remove background
            img_no_bg = self._remove_background(img)
            if self.save_intermediate:
                step1_path = os.path.join(
                    self.intermediate_dir, f"{base_name}_1_bg_removed.png"
                )
                img_no_bg.save(step1_path, "PNG")
                intermediate_paths["bg_removed"] = step1_path

            # Step 2: Remove shadows (optional)
            if self.enable_shadow_removal:
                img_no_shadow = self._remove_shadows(img_no_bg)
                if self.save_intermediate:
                    step2_path = os.path.join(
                        self.intermediate_dir, f"{base_name}_2_no_shadow.png"
                    )
                    img_no_shadow.save(step2_path, "PNG")
                    intermediate_paths["no_shadow"] = step2_path
            else:
                img_no_shadow = img_no_bg

            # Step 3: Smart crop to product boundaries (optional)
            if self.enable_smart_crop:
                img_cropped = self._smart_crop(img_no_shadow)
            else:
                # Simple cropping using basic alpha threshold
                img_cropped = self._simple_crop(img_no_shadow)

            if self.save_intermediate:
                step3_path = os.path.join(
                    self.intermediate_dir, f"{base_name}_3_cropped.png"
                )
                img_cropped.save(step3_path, "PNG")
                intermediate_paths["cropped"] = step3_path

            # Step 4: Resize with aspect ratio preservation
            img_final = self._resize_with_padding(img_cropped)
            if self.save_intermediate:
                step4_path = os.path.join(
                    self.intermediate_dir, f"{base_name}_4_resized.png"
                )
                img_final.save(step4_path, "PNG")
                intermediate_paths["resized"] = step4_path

            # Step 5: Enhance quality
            img_final = self._enhance_image(img_final)

            # Save processed image
            img_final.save(output_path, "PNG", quality=95)

            return {
                "status": "success",
                "original_size": original_size,
                "final_size": img_final.size,
                "output_path": output_path,
                "intermediate_paths": (
                    intermediate_paths if self.save_intermediate else {}
                ),
            }

        except Exception as e:
            return {"status": "error", "error": str(e), "output_path": None}

    def _init_birefnet(self):
        """Initialize BiRefNet model (lazy loading)"""
        if ImageProcessor._birefnet_model is None:
            from transformers import AutoModelForImageSegmentation

            # Set device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load model
            ImageProcessor._birefnet_model = (
                AutoModelForImageSegmentation.from_pretrained(
                    "ZhengPeng7/BiRefNet", trust_remote_code=True
                )
            )
            ImageProcessor._birefnet_model.to(device)
            ImageProcessor._birefnet_model.eval()

            # Set up transform
            ImageProcessor._birefnet_transform = transforms.Compose(
                [
                    transforms.Resize((1024, 1024)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

    def _remove_background_birefnet(self, img: Image.Image) -> Image.Image:
        """
        Remove background using BiRefNet model

        Args:
            img: PIL Image

        Returns:
            Image with transparent background
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Convert to RGB
        img_rgb = img.convert("RGB")
        image_size = img_rgb.size

        # Transform image
        input_images = (
            ImageProcessor._birefnet_transform(img_rgb).unsqueeze(0).to(device)
        )

        # Prediction
        with torch.no_grad():
            preds = ImageProcessor._birefnet_model(input_images)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)

        # Apply mask as alpha channel
        img_rgb.putalpha(mask)

        return img_rgb

    def _remove_background_rmbg_pipeline(self, img: Image.Image) -> Image.Image:
        """
        Remove background using RMBG-1.4 via transformers pipeline

        Args:
            img: PIL Image

        Returns:
            Image with transparent background
        """
        from transformers import pipeline

        # Load pipeline (cached after first use)
        if not hasattr(self, "_rmbg_pipeline"):
            self._rmbg_pipeline = pipeline(
                "image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True
            )

        # Get segmentation result
        result = self._rmbg_pipeline(img)

        # Result is typically the processed image with background removed
        # RMBG-1.4 returns the image directly with transparent background
        if isinstance(result, Image.Image):
            # Ensure RGBA mode
            if result.mode != "RGBA":
                result = result.convert("RGBA")
            return result
        elif isinstance(result, list) and len(result) > 0:
            # If it's a list, take first result
            output = result[0]
            if isinstance(output, dict) and "mask" in output:
                mask = output["mask"]
                img_rgb = img.convert("RGB")

                # Resize mask if needed
                if mask.size != img_rgb.size:
                    mask = mask.resize(img_rgb.size, Image.Resampling.LANCZOS)

                # Convert mask to grayscale
                if mask.mode != "L":
                    mask = mask.convert("L")

                # Apply mask as alpha channel
                img_rgb.putalpha(mask)
                return img_rgb
            else:
                # Return the output as-is
                if output.mode != "RGBA":
                    output = output.convert("RGBA")
                return output
        else:
            # Fallback: return original with full alpha
            img_rgba = img.convert("RGBA")
            return img_rgba

    def _remove_background(self, img: Image.Image) -> Image.Image:
        """
        Remove background using AI-powered model (rembg, BiRefNet local, or BiRefNet API)

        Args:
            img: PIL Image

        Returns:
            Image with transparent background
        """
        # Use RMBG-1.4 if selected
        if self.bg_model == "rmbg-1.4":
            # Use transformers pipeline (works locally, no GPU required for inference)
            return self._remove_background_rmbg_pipeline(img)

        # Otherwise use rembg
        # Create session with selected model
        session = new_session(self.bg_model)

        # Convert to bytes for rembg
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        # Build remove() parameters
        remove_params = {"session": session, "post_process_mask": self.post_process}

        # Add alpha matting parameters if enabled
        if self.alpha_matting:
            remove_params.update(
                {
                    "alpha_matting": True,
                    "alpha_matting_foreground_threshold": 240,
                    "alpha_matting_background_threshold": 10,
                    "alpha_matting_erode_size": 10,
                }
            )

        # Remove background with configured parameters
        output = remove(img_byte_arr, **remove_params)

        # Convert back to PIL Image
        img_no_bg = Image.open(io.BytesIO(output))

        # Ensure RGBA mode
        if img_no_bg.mode != "RGBA":
            img_no_bg = img_no_bg.convert("RGBA")

        return img_no_bg

    def _remove_shadows(self, img: Image.Image) -> Image.Image:
        """
        Detect and remove shadows from the image

        Args:
            img: PIL Image with RGBA mode

        Returns:
            Image with reduced shadows
        """
        # Convert to numpy array
        img_array = np.array(img)

        # Separate alpha channel
        if img_array.shape[2] == 4:
            rgb = img_array[:, :, :3]
            alpha = img_array[:, :, 3]
        else:
            rgb = img_array
            alpha = (
                np.ones((img_array.shape[0], img_array.shape[1]), dtype=np.uint8) * 255
            )

        # Convert to HSV for better shadow detection
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Increase brightness in darker areas (shadow regions)
        h, s, v = cv2.split(hsv)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to brightness
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_enhanced = clahe.apply(v)

        # Merge back
        hsv_enhanced = cv2.merge([h, s, v_enhanced])
        rgb_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

        # Combine with alpha channel
        img_array_enhanced = np.dstack([rgb_enhanced, alpha])

        # Convert back to PIL Image
        img_no_shadow = Image.fromarray(img_array_enhanced, "RGBA")

        return img_no_shadow

    def _smart_crop(self, img: Image.Image) -> Image.Image:
        """
        Intelligently crop to product boundaries using contour detection

        Args:
            img: PIL Image with RGBA mode

        Returns:
            Cropped image
        """
        # Convert to numpy array
        img_array = np.array(img)

        # Get alpha channel
        alpha = img_array[:, :, 3]

        # Apply morphological operations to clean up the alpha channel
        # This removes small artifacts and noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Close small holes in the alpha channel
        alpha_cleaned = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Remove small noise
        alpha_cleaned = cv2.morphologyEx(
            alpha_cleaned, cv2.MORPH_OPEN, kernel, iterations=1
        )

        # Apply threshold to get binary mask
        # Use adaptive threshold to handle varying transparency
        _, binary_mask = cv2.threshold(alpha_cleaned, 25, 255, cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # Fallback to simple alpha detection if no contours found
            rows = np.any(alpha > 10, axis=1)
            cols = np.any(alpha > 10, axis=0)

            if not np.any(rows) or not np.any(cols):
                return img

            row_min, row_max = np.where(rows)[0][[0, -1]]
            col_min, col_max = np.where(cols)[0][[0, -1]]
        else:
            # Find the largest contour (main product)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get bounding rectangle of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)

            col_min, row_min = x, y
            col_max, row_max = x + w, y + h

        # Calculate adaptive padding based on crop tightness
        height, width = img_array.shape[:2]

        # Adjust padding based on tightness setting
        if self.crop_tightness == "tight":
            padding_multiplier = 0.5  # Less padding
        elif self.crop_tightness == "loose":
            padding_multiplier = 1.5  # More padding
        else:  # normal
            padding_multiplier = 1.0

        # Calculate padding
        crop_width = col_max - col_min
        crop_height = row_max - row_min

        padding_h = int(crop_height * self.padding_percent * padding_multiplier)
        padding_w = int(crop_width * self.padding_percent * padding_multiplier)

        # Add minimum padding (at least 10 pixels)
        padding_h = max(10, padding_h)
        padding_w = max(10, padding_w)

        # Apply padding with bounds checking
        row_min = max(0, row_min - padding_h)
        row_max = min(height, row_max + padding_h)
        col_min = max(0, col_min - padding_w)
        col_max = min(width, col_max + padding_w)

        # Crop image
        img_cropped = img.crop((col_min, row_min, col_max, row_max))

        return img_cropped

    def _simple_crop(self, img: Image.Image) -> Image.Image:
        """
        Simple cropping using basic alpha threshold (fallback method)

        Args:
            img: PIL Image with RGBA mode

        Returns:
            Cropped image
        """
        # Convert to numpy array
        img_array = np.array(img)

        # Get alpha channel
        alpha = img_array[:, :, 3]

        # Find non-transparent pixels with simple threshold
        rows = np.any(alpha > 10, axis=1)
        cols = np.any(alpha > 10, axis=0)

        if not np.any(rows) or not np.any(cols):
            return img

        # Find boundaries
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        # Add padding
        height, width = img_array.shape[:2]
        crop_width = col_max - col_min
        crop_height = row_max - row_min

        padding_h = int(crop_height * self.padding_percent)
        padding_w = int(crop_width * self.padding_percent)

        # Apply padding with bounds checking
        row_min = max(0, row_min - padding_h)
        row_max = min(height, row_max + padding_h)
        col_min = max(0, col_min - padding_w)
        col_max = min(width, col_max + padding_w)

        # Crop image
        img_cropped = img.crop((col_min, row_min, col_max, row_max))

        return img_cropped

    def _resize_with_padding(self, img: Image.Image) -> Image.Image:
        """
        Resize image to target size while maintaining aspect ratio
        Add padding to make it square

        Args:
            img: PIL Image

        Returns:
            Resized image with padding
        """
        # Calculate scaling factor
        width, height = img.size
        scale = min(self.target_size / width, self.target_size / height)

        # Resize maintaining aspect ratio
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create new image with target size and background color
        new_img = Image.new(
            "RGB", (self.target_size, self.target_size), self.background_color
        )

        # Calculate position to paste (center)
        paste_x = (self.target_size - new_width) // 2
        paste_y = (self.target_size - new_height) // 2

        # Paste resized image onto background
        new_img.paste(
            img_resized,
            (paste_x, paste_y),
            img_resized if img_resized.mode == "RGBA" else None,
        )

        return new_img

    def _enhance_image(self, img: Image.Image) -> Image.Image:
        """
        Apply subtle enhancements to improve image quality

        Args:
            img: PIL Image

        Returns:
            Enhanced image
        """
        # Slight sharpening
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))

        # Slight contrast enhancement
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.10)

        # Slight color saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.05)

        return img

    def process_batch(self, input_dir: str, output_dir: str) -> list:
        """
        Process multiple images in batch

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save processed images

        Returns:
            List of processing results
        """
        results = []

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Supported image formats
        supported_formats = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        # Process each image
        for filename in os.listdir(input_dir):
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext in supported_formats:
                input_path = os.path.join(input_dir, filename)
                output_filename = f"{os.path.splitext(filename)[0]}_processed.png"
                output_path = os.path.join(output_dir, output_filename)

                result = self.process_image(input_path, output_path)
                result["filename"] = filename
                results.append(result)

        return results


def create_processor(
    target_size: int = 1000,
    padding_percent: float = 5.0,
    crop_tightness: str = "normal",
    enable_shadow_removal: bool = True,
    enable_smart_crop: bool = True,
) -> ImageProcessor:
    """
    Factory function to create an ImageProcessor instance

    Args:
        target_size: Target size for output images
        padding_percent: Percentage of padding around product
        crop_tightness: Cropping mode - 'tight', 'normal', or 'loose'
        enable_shadow_removal: Enable shadow detection and removal
        enable_smart_crop: Enable advanced contour-based cropping

    Returns:
        ImageProcessor instance
    """
    return ImageProcessor(
        target_size=target_size,
        padding_percent=padding_percent,
        crop_tightness=crop_tightness,
        enable_shadow_removal=enable_shadow_removal,
        enable_smart_crop=enable_smart_crop,
    )
