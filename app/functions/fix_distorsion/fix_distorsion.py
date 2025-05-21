import cv2
import numpy as np
import os
import math
from typing import Dict, Tuple, Optional, Any

def detect_distorsion(filepath: str, showSteps: bool = False, scale_factor: Optional[float] = None, 
                     escala_original: Optional[Dict[str, Any]] = None, output_dir: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], np.ndarray, Optional[Dict[str, Any]], Dict[str, str]]:
    """
    Detects distortion and returns adjusted scale information.
    
    Args:
        filepath: Path to the image file
        showSteps: Whether to display intermediate steps
        scale_factor: Scale factor in pixels per unit (optional)
        escala_original: Original scale information (optional)
        output_dir: Directory to save comparison images (optional)
        
    Returns:
        Tuple containing:
        - distortion_info: Dictionary with calculated distortion factors
        - roi: Region of interest (red square)
        - corrected_image: Image with distortion corrected
        - adjusted_scale: Scale adjusted after correction
        - comparison_images: Paths to comparison images
    """
    # Load the image
    image = cv2.imread(filepath)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {filepath}")

    distortion_info, roi, corrected_image, comparison_images = process_distortion_image(image, showSteps, scale_factor, output_dir)
    
    # Calculate adjusted scale if we have the necessary information
    adjusted_scale = None
    if distortion_info is not None and escala_original is not None:
        # Get applied correction factors
        aspect_correction = distortion_info["correction_factor"]
        size_correction = distortion_info.get("size_correction", 1.0)
        
        # Calculate new scale factors
        pixels_per_unit_original = escala_original["pixels_per_unit"]
        
        # Apply corrections to the scale
        horizontal_scale_factor = 1.0 / aspect_correction if aspect_correction < 1.0 else 1.0
        global_scale_factor = size_correction if distortion_info.get("scale_adjusted", False) else 1.0
        
        # Consider rotation adjustment
        rotation_factor = 1.0
        if "rotation_angle" in distortion_info:
            # Scale is not significantly affected by small rotations
            if abs(distortion_info["rotation_angle"]) > 10:
                rotation_factor = 0.98  # Approximate adjustment for large rotations
        
        # Consider perspective adjustment
        perspective_factor = 1.0
        if distortion_info.get("perspective_corrected", False):
            perspective_factor = distortion_info.get("perspective_correction", 1.0)
        
        # Calculate new scale values considering all factors
        pixels_per_unit_adjusted = pixels_per_unit_original * horizontal_scale_factor * global_scale_factor * rotation_factor * perspective_factor
        units_per_pixel_adjusted = 1.0 / pixels_per_unit_adjusted
        
        # Create adjusted scale dictionary
        adjusted_scale = escala_original.copy()
        adjusted_scale.update({
            "pixels_per_unit": pixels_per_unit_adjusted,
            "units_per_pixel": units_per_pixel_adjusted,
            "applied_adjustments": {
                "aspect_correction": aspect_correction,
                "size_correction": size_correction,
                "rotation_factor": rotation_factor,
                "perspective_factor": perspective_factor
            }
        })

        print(f"Aspect correction factor: {aspect_correction:.4f}")
        print(f"Size correction factor: {size_correction:.4f}")
        print(f"Rotation correction factor: {rotation_factor:.4f}")
        print(f"Perspective correction factor: {perspective_factor:.4f}")
        print(f"Adjusted scale: {pixels_per_unit_adjusted:.2f} px/{escala_original['unit']}")
        print("Image with corrected distortion and calculated adjusted scale")
    
    return distortion_info, roi, corrected_image, adjusted_scale, comparison_images

def process_distortion_image(image: np.ndarray, showSteps: bool = False, 
                           scale_factor: Optional[float] = None, 
                           output_dir: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], np.ndarray, Dict[str, str]]:
    """
    Processes an image to detect and correct distortion based on the red square.
    
    Args:
        image: Input image
        showSteps: Whether to display intermediate steps
        scale_factor: Scale factor in pixels per unit (optional)
        output_dir: Directory to save comparison images (optional)
        
    Returns:
        Tuple containing distortion info, ROI, corrected image, and comparison image paths
    """
    # Dictionary to store comparison image paths
    comparison_images = {}
    
    # Create folder for distortion images if it doesn't exist
    distortion_dir = output_dir
    if output_dir:
        distortion_dir = os.path.join(output_dir, "distortion")
        os.makedirs(distortion_dir, exist_ok=True)
    
    # Step 1: Find the red square
    square_roi, square_contour, corners = find_red_square(image, showSteps)
    if square_roi is None:
        print("Red square not detected in the image.")
        return None, None, image, comparison_images
    
    # Save image of red square before correction
    if distortion_dir and square_contour is not None:
        before_square = extract_square_roi(image, square_contour, margin_factor=1.5)
        if before_square is not None:
            before_path = os.path.join(distortion_dir, "before_red_square.png")
            cv2.imwrite(before_path, before_square)
            comparison_images["before_square"] = before_path
    
    # Step 2: Calculate distortion factor using scale if available
    distortion_info = calculate_distortion(square_contour, corners, showSteps, scale_factor)
    if distortion_info is None:
        print("Could not calculate distortion.")
        return None, square_roi, image, comparison_images
    
    # Step 3: Apply distortion correction
    corrected_image = apply_distortion_correction(image, distortion_info, corners, showSteps)
    
    # Step 4: Save image of red square after correction
    if distortion_dir and square_contour is not None:
        # Find the red square in the corrected image
        corrected_roi, corrected_contour, _ = find_red_square(corrected_image, False)
        if corrected_contour is not None:
            after_square = extract_square_roi(corrected_image, corrected_contour, margin_factor=1.5)
            if after_square is not None:
                after_path = os.path.join(distortion_dir, "after_red_square.png")
                cv2.imwrite(after_path, after_square)
                comparison_images["after_square"] = after_path
    
    # Show intermediate steps if requested
    if showSteps:
        cv2.imshow("Original image", image)
        cv2.imshow("Detected square", square_roi)
        cv2.imshow("Corrected result", corrected_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return distortion_info, square_roi, corrected_image, comparison_images

def extract_square_roi(image: np.ndarray, contour: np.ndarray, margin_factor: float = 1.5) -> Optional[np.ndarray]:
    """
    Extracts a region of interest around the red square with additional margin.
    
    Args:
        image: Input image
        contour: Square contour
        margin_factor: Factor to expand the crop (1.5 = 50% larger)
        
    Returns:
        Region of interest cropped around the square
    """
    if contour is None or len(image.shape) < 2:
        return None
    
    # Get the rectangle containing the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate the center of the rectangle
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Calculate square size with margin
    square_size = int(max(w, h) * margin_factor)
    half_size = square_size // 2
    
    # Calculate crop coordinates
    start_x = max(0, center_x - half_size)
    start_y = max(0, center_y - half_size)
    end_x = min(image.shape[1], center_x + half_size)
    end_y = min(image.shape[0], center_y + half_size)
    
    # Extract the region of interest
    roi = image[start_y:end_y, start_x:end_x]
    
    return roi

def find_red_square(image: np.ndarray, showSteps: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Finds the red square in the image.
    
    Args:
        image: Input image
        showSteps: Whether to display intermediate steps
        
    Returns:
        Tuple containing:
        - roi: Region of interest containing the red square
        - contour: Contour of the red square
        - corners: List of 4 points forming the corners of the square
    """
    # Convert to HSV to filter the red color
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define red color ranges in HSV (red can be in two ranges)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if showSteps:
        mask_img = image.copy()
        cv2.drawContours(mask_img, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Red mask", red_mask)
        cv2.imshow("Detected contours", mask_img)
        cv2.waitKey(0)
    
    # Filter very small contours (noise)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    if not valid_contours:
        return None, None, None
    
    # Filter to find squares (contours with approx 4 sides) and sort by Y position
    square_candidates = []
    
    for contour in valid_contours:
        # Approximate contour to a polygon
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If it has 4 sides, it's a candidate to be a square
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            ratio = float(w) / h if h > 0 else float('inf')
            # Only consider if aspect ratio is close to 1 (square)
            if abs(ratio - 1.0) < 0.3:  # Tolerance to consider a square
                square_candidates.append({
                    'contour': contour,
                    'corners': approx,
                    'y': y  # Save Y coordinate for sorting
                })
    
    # If we find squares, select the one that's higher up (smaller Y)
    if square_candidates:
        # Sort by Y coordinate (ascending)
        square_candidates.sort(key=lambda x: x['y'])
        best_contour = square_candidates[0]['contour']
        best_corners = square_candidates[0]['corners']
    # If no valid squares, use the largest contour
    elif valid_contours:
        best_contour = max(valid_contours, key=cv2.contourArea)
        epsilon = 0.04 * cv2.arcLength(best_contour, True)
        best_corners = cv2.approxPolyDP(best_contour, epsilon, True)
        # Ensure it has 4 corners
        if len(best_corners) != 4:
            # Use bounding box corners
            x, y, w, h = cv2.boundingRect(best_contour)
            best_corners = np.array([
                [[x, y]],
                [[x+w, y]],
                [[x+w, y+h]],
                [[x, y+h]]
            ], dtype=np.int32)
    else:
        return None, None, None
    
    # Sort square corners clockwise starting from top-left
    if len(best_corners) == 4:
        best_corners = order_points(best_corners.reshape(4, 2))
    
    # Extract ROI of the red square
    x, y, w, h = cv2.boundingRect(best_contour)
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)
    square_roi = image[y:y+h, x:x+w]
    
    if showSteps:
        contour_img = image.copy()
        cv2.drawContours(contour_img, [best_contour], -1, (0, 255, 0), 2)
        # Draw corners
        if best_corners is not None:
            for i, point in enumerate(best_corners):
                cv2.circle(contour_img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
                cv2.putText(contour_img, str(i), (int(point[0]), int(point[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Detected red square", contour_img)
        cv2.imshow("Square ROI", square_roi)
        cv2.waitKey(0)
    
    return square_roi, best_contour, best_corners

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points clockwise starting from top-left.
    
    Args:
        pts: Array of 4 points [x,y]
        
    Returns:
        Array of 4 ordered points
    """
    # Initialize array for ordered points
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Get sum and difference of coordinates
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    # The point with smallest sum is the top-left corner
    rect[0] = pts[np.argmin(s)]
    # The point with largest sum is the bottom-right corner
    rect[2] = pts[np.argmax(s)]
    
    # The point with smallest difference is the top-right corner
    rect[1] = pts[np.argmin(diff)]
    # The point with largest difference is the bottom-left corner
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def calculate_distortion(contour: np.ndarray, corners: np.ndarray, 
                        showSteps: bool = False, scale_factor: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Calculates distortion based on square dimensions and its known size of 2x2cm.
    
    Args:
        contour: Contour of the red square
        corners: Corners of the red square
        showSteps: Whether to display intermediate steps
        scale_factor: Scale factor in pixels per unit (optional)
        
    Returns:
        Dictionary with distortion information
    """
    if contour is None:
        return None
    
    # Get contour dimensions
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calculate aspect ratio (should be 1 for a square)
    aspect_ratio = float(w) / h if h > 0 else 1.0
    
    # Calculate rotation angle
    rotation_angle = 0
    if corners is not None and len(corners) == 4:
        # Use the top two corners to calculate angle
        x1, y1 = corners[0]
        x2, y2 = corners[1]
        # Calculate rotation angle in degrees
        if x2 != x1:
            angle_rad = math.atan2(y2 - y1, x2 - x1)
            rotation_angle = math.degrees(angle_rad)
            # Normalize to [-45, 45] degrees
            if rotation_angle > 45:
                rotation_angle -= 90
            elif rotation_angle < -45:
                rotation_angle += 90
    
    # Detect perspective
    has_perspective = False
    perspective_correction = 1.0
    if corners is not None and len(corners) == 4:
        # Calculate proportions to detect perspective distortion
        top_width = np.linalg.norm(corners[1] - corners[0])
        bottom_width = np.linalg.norm(corners[2] - corners[3])
        left_height = np.linalg.norm(corners[3] - corners[0])
        right_height = np.linalg.norm(corners[2] - corners[1])
        
        # If there are significant differences between these values, there's perspective
        width_ratio = max(top_width, bottom_width) / min(top_width, bottom_width)
        height_ratio = max(left_height, right_height) / min(left_height, right_height)
        
        if width_ratio > 1.05 or height_ratio > 1.05:
            has_perspective = True
            perspective_correction = (width_ratio + height_ratio) / 2
    
    # Use scale information to compare with the reference 2x2cm square
    scale_adjusted = False
    size_correction = 1.0
    
    if scale_factor is not None:
        # Expected size for a 2x2cm square in pixels
        expected_size_px = 20 * scale_factor  # 2cm = 20mm
        
        # Calculate size correction factor
        current_size_avg = (w + h) / 2  # Average of width and height
        size_correction = expected_size_px / current_size_avg
        scale_adjusted = True
        
        if showSteps:
            print(f"Scale factor: {scale_factor:.2f} px/mm")
            print(f"Expected size for 2x2cm: {expected_size_px:.1f}px")
            print(f"Current average size: {current_size_avg:.1f}px")
            print(f"Size correction factor: {size_correction:.4f}")
            print(f"Rotation angle: {rotation_angle:.2f} degrees")
            if has_perspective:
                print(f"Perspective distortion detected: {perspective_correction:.4f}")
    
    # Distortion information
    distortion_info = {
        "aspect_ratio": aspect_ratio,
        "correction_factor": aspect_ratio,  # Aspect ratio correction factor
        "size_correction": size_correction,  # Size correction factor based on scale
        "width_pixels": w,
        "height_pixels": h,
        "reference_size_cm": 2.0,  # Reference square is 2x2 cm
        "scale_adjusted": scale_adjusted,
        "rotation_angle": rotation_angle,
        "has_perspective": has_perspective,
        "perspective_correction": perspective_correction,
        "perspective_corrected": has_perspective,
        "corners": corners if corners is not None else []
    }
    
    return distortion_info

def apply_distortion_correction(image: np.ndarray, distortion_info: Dict[str, Any], 
                              corners: np.ndarray, showSteps: bool = False) -> np.ndarray:
    """
    Applies distortion correction to the image considering aspect ratio, size, rotation, and perspective.
    
    Args:
        image: Original image
        distortion_info: Information about the distortion
        corners: Square corners for perspective
        showSteps: Whether to display intermediate steps
        
    Returns:
        Image with corrected distortion
    """
    if distortion_info is None:
        return image
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Step 1: Correct perspective if necessary
    corrected_image = image.copy()
    
    if distortion_info["has_perspective"] and corners is not None and len(corners) == 4:
        if showSteps:
            print("Applying perspective correction...")
        
        # Calculate corrected square side (average of current sides)
        current_sides = [
            np.linalg.norm(corners[1] - corners[0]),  # top
            np.linalg.norm(corners[2] - corners[1]),  # right
            np.linalg.norm(corners[2] - corners[3]),  # bottom
            np.linalg.norm(corners[3] - corners[0])   # left
        ]
        target_side = np.mean(current_sides)
        
        # Define destination points for a perfect square
        dst_points = np.array([
            [0, 0],
            [target_side, 0],
            [target_side, target_side],
            [0, target_side]
        ], dtype=np.float32)
        
        # Convert corners to proper format
        src_points = np.array(corners, dtype=np.float32)
        
        # Calculate transformation matrix and apply it
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        corrected_square = cv2.warpPerspective(image, matrix, (int(target_side), int(target_side)))
        
        if showSteps:
            cv2.imshow("Square with corrected perspective", corrected_square)
            cv2.waitKey(0)
        
        # Scale the corrected square to appropriate size to maintain proportions
        scale_to_original = min(width, height) / target_side
        new_size = (int(width / scale_to_original), int(height / scale_to_original))
        
        # Transform the entire image
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points * scale_to_original)
        corrected_image = cv2.warpPerspective(image, perspective_matrix, (width, height))
    
    # Step 2: Correct rotation if necessary
    if abs(distortion_info["rotation_angle"]) > 1.0:  # 1 degree threshold
        if showSteps:
            print(f"Applying rotation correction: {distortion_info['rotation_angle']:.2f} degrees")
        
        # Get image center
        center = (width / 2, height / 2)
        
        # Calculate rotation matrix and apply it
        rotation_matrix = cv2.getRotationMatrix2D(center, -distortion_info["rotation_angle"], 1.0)
        rotated_image = cv2.warpAffine(corrected_image, rotation_matrix, (width, height), 
                                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        corrected_image = rotated_image
    
    # Step 3: Correct aspect ratio
    aspect_correction = distortion_info["correction_factor"]
    
    if abs(aspect_correction - 1.0) > 0.01:  # Only correct if there's significant difference
        if showSteps:
            print(f"Applying aspect ratio correction: {aspect_correction:.4f}")
        
        if aspect_correction > 1.0:
            # Image too wide, correct height
            new_height = int(height * aspect_correction)
            new_width = width
        else:
            # Image too tall, correct width
            new_height = height
            new_width = int(width / aspect_correction)
        
        # Apply aspect ratio correction
        corrected_image = cv2.resize(corrected_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Step 4: Apply size correction if necessary
    size_correction = distortion_info.get("size_correction", 1.0)
    
    if size_correction != 1.0 and distortion_info.get("scale_adjusted", False):
        if showSteps:
            print(f"Applying size correction: {size_correction:.4f}")
        
        final_height = int(corrected_image.shape[0] * size_correction)
        final_width = int(corrected_image.shape[1] * size_correction)
        corrected_image = cv2.resize(corrected_image, (final_width, final_height), 
                                    interpolation=cv2.INTER_CUBIC)
    
    if showSteps:
        print(f"Original dimensions: {width}x{height}")
        print(f"Final dimensions: {corrected_image.shape[1]}x{corrected_image.shape[0]}")
        
        cv2.imshow("Original image", image)
        cv2.imshow("Corrected image", corrected_image)
        cv2.waitKey(0)
    
    return corrected_image