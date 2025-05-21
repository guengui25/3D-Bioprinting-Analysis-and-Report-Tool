import cv2
import numpy as np
import os
import re
import pytesseract
from typing import Tuple, Optional

def extract_red_mask(input_image_path: str, output_dir: Optional[str] = None, 
                    showSteps: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts red regions from an image and creates a binary mask.
    
    Args:
        input_image_path: Path to the input image
        output_dir: Directory to save output images
        showSteps: Whether to show intermediate steps
        
    Returns:
        Tuple containing the red mask and the original image
    """
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_image_path}")
    
    # Convert to HSV (better for color segmentation)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for red color (in HSV red is split)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create and combine masks
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean the mask with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    if showSteps:
        cv2.imshow('Red Mask', red_mask)
        cv2.waitKey(0)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, "red_mask.png"), red_mask)
    
    return red_mask, image

def extract_scale_from_image(input_path: str, output_dir: Optional[str] = None, 
                            showSteps: bool = False, double_scale: bool = False) -> Tuple[Optional[float], str]:
    """
    Extracts scale information from an image.
    
    Args:
        input_path: Path to the input image
        output_dir: Directory to save output images
        showSteps: Whether to show intermediate steps
        double_scale: Whether to double the calculated scale
        
    Returns:
        Tuple containing the scale factor (pixels per mm) and the extracted OCR text
    """
    # Extract red regions
    red_mask, original_image = extract_red_mask(input_path, output_dir, showSteps)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Look for the horizontal bar (likely the scale bar)
    scale_bar = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Look for horizontal lines (width >> height)
        if aspect_ratio > 5 and w > 20:
            # Sort by Y position (the lowest is typically the scale bar)
            if scale_bar is None or y + h > scale_bar[1] + scale_bar[3]:
                scale_bar = (x, y, w, h)
    
    if not scale_bar:
        return None, "No scale bar found"
    
    x, y, w, h = scale_bar
    
    # Apply mask to the original image to get red components
    masked_image = cv2.bitwise_and(original_image, original_image, mask=red_mask)
    
    # Convert to grayscale and binarize
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Perform OCR to extract text
    try:
        ocr_text = pytesseract.image_to_string(binary, config='--psm 11')
    except Exception as e:
        return None, f"OCR Error: {e}"
    
    # Extract numeric value using regex (look for digits followed by "mm")
    scale_match = re.search(r'(\d+)(?:\s*)(?:mm)', ocr_text, re.IGNORECASE)
    
    if not scale_match:
        return None, f"No scale information found in text: '{ocr_text}'"
    
    scale_value = int(scale_match.group(1))
    
    # Calculate pixels per mm
    pixels_per_mm = w / scale_value

    if double_scale:
        pixels_per_mm = pixels_per_mm * 2
    
    if showSteps:
        display_image = original_image.copy()
        cv2.rectangle(display_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imshow(f'Scale bar: {w} pixels = {scale_value} mm', display_image)
        cv2.waitKey(0)
    
    if output_dir:
        display_image = original_image.copy()
        cv2.rectangle(display_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(output_dir, "scale_bar_detected.png"), display_image)
    
    return pixels_per_mm, ocr_text

if __name__ == "__main__":
    gcode_image_path = "your_gcode_image_path_here.png"  # Replace with your image path if you want to test
    output_dir = "your_output_directory_here"  # Replace with your output directory if you want to test

    os.makedirs(output_dir, exist_ok=True)
    
    scale_factor, scale_text = extract_scale_from_image(
        gcode_image_path, 
        output_dir, 
        showSteps=False
    )
    
    if scale_factor:
        print(f"Detected scale: {scale_factor:.4f} pixels/mm")
        print(f"Extracted text: {scale_text}")
    else:
        print("Could not determine scale")