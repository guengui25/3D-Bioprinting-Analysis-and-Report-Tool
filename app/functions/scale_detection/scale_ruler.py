import cv2
import numpy as np
import pytesseract
import re
from typing import Dict, Tuple, List, Optional, Any

# Default values
DEFAULT_MAX_SCALE = 50.0  # Maximum scale in mm
DEFAULT_UNIT = "mm"       # Default measurement unit

def detect_ruler_scale(filepath: str, showSteps: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detects the ruler region of interest and performs OCR to determine the scale.
    
    Args:
        filepath: Path to the image file
        showSteps: Whether to display intermediate processing steps
        
    Returns:
        Tuple containing:
        - scale_info: Dictionary with scale information
        - roi: Image of the region of interest with measurement
        - result_image: Result image with annotations
    """
    # Load the image
    image = cv2.imread(filepath)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {filepath}")

    scale_info, roi, result_image = process_ruler_image(image, showSteps=showSteps)
    
    return scale_info, roi, result_image

def process_ruler_image(image: np.ndarray, showSteps: bool = False) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Processes an image to detect the ruler and determine the scale.
    
    Args:
        image: Input image array
        showSteps: Whether to display intermediate processing steps
        
    Returns:
        Tuple containing scale info, ruler ROI, and result image
    """
    # Step 1: Detect ruler in the image
    roi_ruler, ruler_contour, _ = find_blue_ruler(image, showSteps=showSteps)
    if roi_ruler is None:
        return None, None, None
    
    # Step 2: Get ROI coordinates
    x_roi, y_roi, w_roi, h_roi = get_roi_coordinates(ruler_contour)
    
    # Step 3: Process ROI to detect markings
    ruler_with_measurement, ruler_width, _, top_left, top_right = process_ruler_markings(roi_ruler, showSteps=showSteps)
    if ruler_with_measurement is None:
        return None, None, None
    
    # Step 4: Extract and calculate scale using OCR
    min_val, max_val, unit = extract_scale_values_with_ocr(roi_ruler, showSteps=showSteps)
    scale_info = calculate_scale_factors(ruler_width, min_val, max_val, unit)
    
    print(f"Scale: {scale_info['units_per_pixel']:.6f} {unit}/px ({scale_info['pixels_per_unit']:.2f} px/{unit})")
    print(f"Detected range: {scale_info['unit_range']} {unit}")
    
    # Step 5: Transform measurement line coordinates
    line_points = transform_coordinates_to_original_image((top_left, top_right), x_roi, y_roi, w_roi, h_roi)
    
    # Step 6: Create result image with annotations
    result_image = create_result_image(image, scale_info, ruler_contour, line_points)
    
    # Show intermediate steps if requested
    if showSteps:
        cv2.imshow("Result", result_image)
        cv2.imshow("Ruler with measurement", ruler_with_measurement)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return scale_info, ruler_with_measurement, result_image

def get_roi_coordinates(contour: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Gets the ROI coordinates from the contour in the original image.
    
    Args:
        contour: Contour of the detected ruler
        
    Returns:
        Tuple containing x, y, width, and height of the ROI
    """
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        margin = 5
        x = max(0, x - margin)
        y = max(0, y - margin)
        return x, y, w, h
    return 0, 0, 0, 0

def transform_coordinates_to_original_image(points: Tuple[Tuple[int, int], Tuple[int, int]], 
                                           x_roi: int, y_roi: int, w_roi: int, h_roi: int) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Transforms coordinates from the cropped ROI back to the original image.
    
    Args:
        points: Tuple containing top_left and top_right points
        x_roi, y_roi, w_roi, h_roi: ROI coordinates in the original image
        
    Returns:
        Tuple containing transformed coordinates
    """
    top_left, top_right = points
    
    if top_left is None or top_right is None:
        return None, None
    
    # Calculate offset due to cropping in process_ruler_markings
    crop_x = int(w_roi * 0.03)
    crop_y = int(h_roi * 0.1)
    
    # Transform coordinates to original image
    global_top_left = (int(x_roi + crop_x + top_left[0]), int(y_roi + crop_y + top_left[1]))
    global_top_right = (int(x_roi + crop_x + top_right[0]), int(y_roi + crop_y + top_right[1]))
    
    return global_top_left, global_top_right

def create_result_image(image: np.ndarray, scale_info: Dict[str, Any], contour: np.ndarray, 
                       line_points: Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]) -> np.ndarray:
    """
    Creates a result image with visual annotations.
    
    Args:
        image: Original input image
        scale_info: Dictionary with scale information
        contour: Ruler contour
        line_points: Points for measurement line
        
    Returns:
        Annotated result image
    """
    # Create a copy of the original image
    original_image = image.copy()
    height, width = original_image.shape[:2]
    
    # If no contour, return the original image
    if contour is None:
        return original_image
    
    # Get rectangle enclosing the ruler contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Define margins around the ROI
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.4)
    
    # Calculate new dimensions and coordinates of ROI with margin
    roi_x1 = max(0, x - margin_x)
    roi_y1 = max(0, y - margin_y)
    roi_x2 = min(width, x + w + margin_x)
    roi_y2 = min(height, y + h + margin_y)
    roi_width = roi_x2 - roi_x1
    roi_height = roi_y2 - roi_y1
    
    # Create blank image for the result
    result = np.ones((roi_height, roi_width, 3), dtype=np.uint8) * 255
    
    # Copy ROI from original image
    roi = original_image[roi_y1:roi_y2, roi_x1:roi_x2]
    result[0:roi_height, 0:roi_width] = roi
    
    # Adjust contour coordinates relative to new ROI
    adjusted_contour = contour.copy()
    adjusted_contour[:, :, 0] = adjusted_contour[:, :, 0] - roi_x1
    adjusted_contour[:, :, 1] = adjusted_contour[:, :, 1] - roi_y1
    
    # Draw ruler contour
    cv2.drawContours(result, [adjusted_contour], -1, (0, 0, 255), 2)
    
    # Draw measurement line if points are available
    if line_points[0] is not None and line_points[1] is not None:
        p1_adjusted = (line_points[0][0] - roi_x1, line_points[0][1] - roi_y1)
        p2_adjusted = (line_points[1][0] - roi_x1, line_points[1][1] - roi_y1)
        cv2.line(result, p1_adjusted, p2_adjusted, (0, 0, 255), 2)
    
    return result

def find_blue_ruler(image: np.ndarray, showSteps: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Filters the image and finds the blue ruler.
    
    Args:
        image: Input image
        showSteps: Whether to display intermediate steps
        
    Returns:
        Tuple containing ruler ROI, ruler contour, and blue mask
    """
    # Get the top half of the image
    height, width = image.shape[:2]
    top_half = image[:height//2, :]
    
    # Convert to HSV and apply blue filter
    hsv = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([150, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    filtered_image = cv2.bitwise_and(top_half, top_half, mask=blue_mask)
    
    if showSteps:
        cv2.imshow("Top half of image", top_half)
        cv2.imshow("Image with blue filter", filtered_image)
        cv2.imshow("Blue mask", blue_mask)
        cv2.waitKey(0)
    
    # Process to get contours
    gray_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if showSteps:
        cv2.imshow("Binary image", binary)
        cv2.waitKey(0)
    
    # Filter out small contours (noise)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
    if not valid_contours:
        return None, None, blue_mask
    
    # Sort by Y coordinate (to get the one closest to the top)
    valid_contours.sort(key=lambda x: cv2.boundingRect(x)[1])
    ruler_contour = valid_contours[0]
    
    # Visualization if needed
    if showSteps:
        contour_img = top_half.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Detected contours", contour_img)
        
        selected_contour = top_half.copy()
        cv2.drawContours(selected_contour, [ruler_contour], -1, (0, 0, 255), 2)
        cv2.imshow("Selected ruler contour", selected_contour)
        cv2.waitKey(0)
    
    # Extract ROI based on contour
    x, y, w, h = cv2.boundingRect(ruler_contour)
    margin = 5
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(width - x, w + 2 * margin)
    h = min(height//2 - y, h + 2 * margin)
    ruler_roi = top_half[y:y+h, x:x+w]
    
    if showSteps and ruler_roi is not None:
        cv2.imshow("Ruler ROI", ruler_roi)
        cv2.waitKey(0)
    
    return ruler_roi, ruler_contour, blue_mask

def process_ruler_markings(ruler_roi: np.ndarray, showSteps: bool = False) -> Tuple[Optional[np.ndarray], float, Optional[List[np.ndarray]], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Processes the ruler image to crop edges and detect divisions.
    
    Args:
        ruler_roi: ROI image containing the ruler
        showSteps: Whether to display intermediate steps
        
    Returns:
        Tuple containing processed image, ruler width, contours, and measurement points
    """
    if ruler_roi is None:
        print("Error: No valid ROI provided")
        return None, 0, None, None, None
    
    # Crop the edges
    height, width = ruler_roi.shape[:2]
    crop_x = int(width * 0.03)
    crop_y = int(height * 0.1)
    
    # Avoid excessive cropping
    if crop_x * 2 >= width or crop_y * 2 >= height:
        crop_x = min(crop_x, width // 10)
        crop_y = min(crop_y, height // 10)
    
    ruler_content = ruler_roi[crop_y:height-crop_y, crop_x:width-crop_x]
    
    if showSteps:
        cv2.imshow("Original ruler ROI", ruler_roi)
        cv2.imshow("Ruler without edges", ruler_content)
        cv2.waitKey(0)
    
    # Prepare binary image
    gray = cv2.cvtColor(ruler_content, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    if showSteps:
        cv2.imshow("Binary ruler image", binary)
        cv2.waitKey(0)
    
    # Detect contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ruler_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 5]
    
    # Create image for visualization
    contour_img = ruler_content.copy()
    cv2.drawContours(contour_img, ruler_contours, -1, (0, 255, 0), 1)
    
    # Default value for width
    ruler_width = width - (2 * crop_x)
    
    # Find lowest contour
    if not ruler_contours:
        return contour_img, ruler_width, ruler_contours, None, None
    
    sorted_contours = sorted(ruler_contours, key=lambda cnt: max([p[0][1] for p in cnt]))
    bottom_contour = sorted_contours[-1]
    
    # Find measurement points
    if bottom_contour is None:
        return contour_img, ruler_width, ruler_contours, None, None
        
    # Find extreme points
    leftmost = tuple(bottom_contour[bottom_contour[:, :, 0].argmin()][0])
    rightmost = tuple(bottom_contour[bottom_contour[:, :, 0].argmax()][0])
    
    # Look for top points at both extremes
    left_points = [p[0] for p in bottom_contour if p[0][0] <= leftmost[0] + 5]
    right_points = [p[0] for p in bottom_contour if p[0][0] >= rightmost[0] - 5]
    
    if not left_points or not right_points:
        return contour_img, ruler_width, ruler_contours, None, None
        
    top_left = min(left_points, key=lambda p: p[1])
    top_right = min(right_points, key=lambda p: p[1])
    
    # Draw measurement line
    p1 = (int(top_left[0]), int(top_left[1]))
    p2 = (int(top_right[0]), int(top_right[1]))
    cv2.line(contour_img, p1, p2, (0, 0, 255), 2)
    measured_width = np.sqrt((top_right[0] - top_left[0])**2 + (top_right[1] - top_left[1])**2)
    
    # Update width if measurement was successful
    if measured_width > 0:
        ruler_width = measured_width
        
        if showSteps:
            print(f"Measured ruler width: {ruler_width:.2f} pixels")
            print(f"Measurement points: {top_left} to {top_right}")
    
    if showSteps:
        cv2.imshow("Detected ruler shape", contour_img)
        cv2.waitKey(0)
    
    return contour_img, ruler_width, ruler_contours, top_left, top_right

def extract_scale_values_with_ocr(roi: np.ndarray, showSteps: bool = False) -> Tuple[float, float, str]:
    """
    Extracts scale values using OCR in the region of interest.
    
    Args:
        roi: Region of interest
        showSteps: Whether to display intermediate steps
        
    Returns:
        Tuple containing minimum value, maximum value, and unit
    """
    if roi is None:
        return 0.0, DEFAULT_MAX_SCALE, DEFAULT_UNIT
    
    # Preprocess image for OCR
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    
    if showSteps:
        cv2.imshow("ROI for OCR", roi)
        cv2.imshow("Preprocessed ROI", thresh)
        cv2.waitKey(0)
    
    # Perform OCR
    ocr_text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 11')
    
    if showSteps:
        print(f"Original OCR text: {ocr_text}")
    
    # Correct common errors
    # Replace 'O' or 'o' followed by 'mm' with '0mm'
    ocr_text = re.sub(r'[Oo](\s*mm)', r'0\1', ocr_text)
    # Replace 'l' followed by 'mm' with '1mm'
    ocr_text = re.sub(r'l(\s*mm)', r'1\1', ocr_text)
    # Clean text
    ocr_text = re.sub(r'[^0-9.\s\-mcinMCIN]', '', ocr_text)
    
    if showSteps:
        print(f"Corrected OCR text: {ocr_text}")
    
    # Look for complete measurements (number + unit)
    measurements = re.findall(r'(\d+(?:\.\d+)?)\s*(mm|cm|m|in|inch)', ocr_text.lower())
    
    # Determine unit
    unit_matches = re.findall(r'(?:mm|cm|m|in|inch)', ocr_text.lower())
    unit = DEFAULT_UNIT
    if unit_matches:
        unit = unit_matches[0]
        if unit in ['in', 'inch']:
            unit = 'in'
    
    # Extract numerical values
    numbers = []
    if measurements:
        numbers = [float(value) for value, unit_str in measurements]
    else:
        # Look for numbers
        matches = re.findall(r'\d+(?:\.\d+)?', ocr_text)
        for match in matches:
            try:
                val = float(match)
                if val < 1000:
                    numbers.append(val)
            except ValueError:
                pass
    
    # Determine min/max
    min_val, max_val = 0.0, DEFAULT_MAX_SCALE
    
    if numbers:
        numbers.sort()
        if len(numbers) >= 2:
            min_val, max_val = numbers[0], numbers[-1]
        elif len(numbers) == 1:
            if numbers[0] < 0.1:
                min_val, max_val = 0.0, DEFAULT_MAX_SCALE
            else:
                min_val, max_val = 0.0, numbers[0]
    
    # Ensure 0 is minimum if present
    if 0.0 in numbers and len(numbers) > 1:
        non_zero_values = [v for v in numbers if v > 0]
        if non_zero_values:
            min_val = 0.0
            max_val = max(non_zero_values)
    
    # Verify coherence
    if min_val >= max_val:
        min_val = 0.0
    
    if showSteps:
        print(f"Detected values: {numbers}")
        print(f"Detected unit: {unit}")
        print(f"Min: {min_val}, Max: {max_val}")
    
    return min_val, max_val, unit

def calculate_scale_factors(contour_width: float, min_val: float, max_val: float, unit: str = "mm") -> Dict[str, Any]:
    """
    Calculates scale factors based on contour width and detected values.
    
    Args:
        contour_width: Width of the contour in pixels
        min_val: Minimum value detected
        max_val: Maximum value detected
        unit: Measurement unit
        
    Returns:
        Dictionary with scale information
    """
    if max_val <= min_val:
        print("Error: maximum value must be greater than minimum value")
        return {
            "units_per_pixel": 0,
            "pixels_per_unit": 0,
            "unit_range": 0,
            "contour_width_pixels": contour_width,
            "unit": unit
        }
    
    unit_range = max_val - min_val
    units_per_pixel = unit_range / contour_width
    pixels_per_unit = contour_width / unit_range
    
    return {
        "units_per_pixel": units_per_pixel,
        "pixels_per_unit": pixels_per_unit,
        "unit_range": unit_range,
        "contour_width_pixels": contour_width,
        "unit": unit
    }

if __name__ == "__main__":
    filepath = "your_image_path_here.jpg"  # Replace with your image path if you want to test
    scale_info, roi, result = detect_ruler_scale(filepath=filepath, showSteps=False)
    if result is not None:
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()