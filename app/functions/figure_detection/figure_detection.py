import cv2
import numpy as np
import sys
from typing import Tuple, List, Optional

def segment_3d_figure(image: np.ndarray, showSteps: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Segments the image to find the 3D printed figure.
    
    Args:
        image: Input image (corrected for scale and distortion)
        showSteps: Whether to display intermediate steps
        
    Returns:
        Tuple containing:
        - segmented_image: Image with the ROI and all found contours drawn on it
        - contours: List of all contours found inside the ROI 
          (largest one first, followed by internal contours)
    """
    height, width = image.shape[:2]
    
    # Step 1: Use the largest contour in the image as the ROI candidate
    roi = find_largest_contour_roi(image, showSteps)
    if roi is None:
        center_x = width // 2
        center_y = height // 2
        roi_size = min(width, height) // 3
        roi = (center_x - roi_size // 2, center_y - roi_size // 2, roi_size, roi_size)
        print("Using default center region as ROI")
    
    x, y, w, h = roi
    x, y = max(0, x), max(0, y)
    w, h = min(width - x, w), min(height - y, h)
    
    # Extract ROI of the image
    roi_image = image[y:y+h, x:x+w]
    
    if showSteps:
        cv2.imshow("ROI", roi_image)
        cv2.waitKey(0)
    
    # Step 2: Find all contours inside the ROI
    contours = find_centered_contour(roi_image, showSteps)
    
    # Adjust the coordinates of all contours to be relative to the original image
    adjusted_contours = adjust_contour_coordinates(contours, x, y)
    
    # Create a visualization image drawing the ROI and all found contours
    segmented_image = create_segmentation_visualization(image, x, y, w, h, adjusted_contours)
    
    if showSteps:
        cv2.imshow("Final Segmentation", segmented_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return segmented_image, adjusted_contours

def adjust_contour_coordinates(contours: List[np.ndarray], x_offset: int, y_offset: int) -> List[np.ndarray]:
    """
    Adjusts contour coordinates to be relative to the original image.
    
    Args:
        contours: List of contours
        x_offset: X-coordinate offset
        y_offset: Y-coordinate offset
        
    Returns:
        List of adjusted contours
    """
    adjusted_contours = []
    for contour in contours:
        adjusted_contour = contour.copy()
        adjusted_contour[:, :, 0] += x_offset  # adjust x coordinate
        adjusted_contour[:, :, 1] += y_offset  # adjust y coordinate
        adjusted_contours.append(adjusted_contour)
    return adjusted_contours

def create_segmentation_visualization(image: np.ndarray, x: int, y: int, w: int, h: int, 
                                     contours: List[np.ndarray]) -> np.ndarray:
    """
    Creates a visualization image with ROI and contours.
    
    Args:
        image: Original image
        x, y, w, h: ROI coordinates
        contours: List of contours
        
    Returns:
        Image with visualization
    """
    segmented_image = image.copy()
    cv2.rectangle(segmented_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Draw all contours with different colors
    for i, contour in enumerate(contours):
        color = (0, 0, 255) if i == 0 else (0, 255-min(i*30, 255), min(i*30, 255))
        cv2.drawContours(segmented_image, [contour], -1, color, 2)
    
    return segmented_image

def find_largest_contour_roi(image: np.ndarray, showSteps: bool = False) -> Optional[Tuple[int, int, int, int]]:
    """
    Finds the largest contour near the center of the image.
    
    Args:
        image: Input image
        showSteps: Whether to display intermediate steps
        
    Returns:
        Tuple of (x, y, w, h) coordinates of the bounding box or None if not found
    """
    height, width = image.shape[:2]
    image_center = (width // 2, height // 2)
    
    # Get edges from the image
    edges = detect_image_edges(image, showSteps)
    
    if showSteps:
        cv2.imshow("Dilated Edges", edges)
        cv2.waitKey(0)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Visualization for debugging
    if showSteps:
        contour_image = image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        cv2.imshow("All Contours", contour_image)
        cv2.waitKey(0)
    
    # Filter small contours and those far from center
    valid_contours = filter_contours_by_center_proximity(contours, image_center, width)
    
    # If we have valid contours, choose the largest one
    if valid_contours:
        # Sort by area (largest first)
        valid_contours.sort(key=lambda c: c[4], reverse=True)
        
        # Take the largest contour
        largest_contour = valid_contours[0]
        
        if showSteps:
            largest_contour_image = image.copy()
            x, y, w, h = largest_contour[:4]
            cv2.rectangle(largest_contour_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.imshow("Largest Contour", largest_contour_image)
            cv2.waitKey(0)
            
        return largest_contour[:4]  # Return (x, y, w, h)
    
    return None

def detect_image_edges(image: np.ndarray, showSteps: bool = False) -> np.ndarray:
    """
    Detects edges in the image with preprocessing.
    
    Args:
        image: Input image
        showSteps: Whether to display intermediate steps
        
    Returns:
        Edge image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Apply dilation to connect edge gaps
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    if showSteps:
        cv2.imshow("Grayscale", gray)
        cv2.imshow("Edges", edges)
        cv2.waitKey(0)
    
    return dilated

def filter_contours_by_center_proximity(contours: List[np.ndarray], image_center: Tuple[int, int], 
                                       width: int) -> List[Tuple[int, int, int, int, float, float]]:
    """
    Filters contours based on size and distance from image center.
    
    Args:
        contours: List of contours
        image_center: Center point of the image (x, y)
        width: Width of the image
        
    Returns:
        List of filtered contours with metadata
    """
    valid_contours = []
    for contour in contours:
        # Skip very small contours
        if cv2.contourArea(contour) < 1000:
            continue
            
        # Calculate center of contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # Fallback for degenerate contours
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            
        # Calculate distance to image center
        distance_to_center = np.sqrt((cx - image_center[0])**2 + (cy - image_center[1])**2)
        
        # Check if reasonably centered (within 30% of image width from center)
        if distance_to_center < width * 0.3:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            valid_contours.append((x, y, w, h, cv2.contourArea(contour), distance_to_center))
    
    return valid_contours

def find_centered_contour(roi_image: np.ndarray, showSteps: bool = False) -> List[np.ndarray]:
    """
    Finds the largest central contour and all contours inside it.
    
    Args:
        roi_image: ROI image
        showSteps: Whether to display intermediate steps
        
    Returns:
        List of contours found in the ROI (largest one first, followed by internal contours)
    """
    # Get ROI dimensions and create a smaller sub-ROI
    roi_height, roi_width = roi_image.shape[:2]
    sub_roi_image, sub_roi_x, sub_roi_y, sub_roi_width, sub_roi_height = extract_central_sub_roi(
        roi_image, roi_width, roi_height
    )
    
    if showSteps:
        cv2.imshow("Sub-ROI Content", sub_roi_image)
        cv2.waitKey(0)
    
    # Process image to get binary threshold
    thresh = preprocess_image_for_contour_detection(sub_roi_image, showSteps)
    
    if showSteps:
        cv2.imshow("Thresholded Image", thresh)
        cv2.waitKey(0)
    
    # Find contours with hierarchy to detect inner contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Visualization of all detected contours
    if showSteps and contours:
        contour_vis = sub_roi_image.copy()
        for i, c in enumerate(contours):
            if cv2.contourArea(c) < 600:  # Skip tiny contours
                continue
            cv2.drawContours(contour_vis, [c], -1, (0, 255, 0), 2)
        
        cv2.imshow("All Detected Contours", contour_vis)
        cv2.waitKey(0)
    
    # Process contours to find primary and nested contours
    sub_roi_center = (sub_roi_width // 2, sub_roi_height // 2)
    central_contours = filter_contours_by_quality(contours, sub_roi_center, sub_roi_width, sub_roi_height)
    
    # Find primary contour and any nested contours
    filtered_contours = find_primary_and_nested_contours(
        central_contours, contours, sub_roi_image, sub_roi_x, sub_roi_y, 
        sub_roi_width, sub_roi_height, sub_roi_center, showSteps
    )
    
    return filtered_contours

def extract_central_sub_roi(roi_image: np.ndarray, roi_width: int, roi_height: int) -> Tuple[np.ndarray, int, int, int, int]:
    """
    Extracts a central sub-ROI from the ROI image.
    
    Args:
        roi_image: ROI image
        roi_width: Width of ROI
        roi_height: Height of ROI
        
    Returns:
        Tuple containing sub-ROI image and its coordinates
    """
    # Create a smaller sub-ROI (reduced by 30% in both dimensions)
    reduction_percent = 0.3
    sub_roi_width = int(roi_width * (1 - reduction_percent))
    sub_roi_height = int(roi_height * (1 - reduction_percent))
    
    # Calculate sub-ROI coordinates (centered)
    sub_roi_x = (roi_width - sub_roi_width) // 2
    sub_roi_y = (roi_height - sub_roi_height) // 2
    
    # Extract sub-ROI
    sub_roi_image = roi_image[sub_roi_y:sub_roi_y+sub_roi_height, sub_roi_x:sub_roi_x+sub_roi_width]
    
    return sub_roi_image, sub_roi_x, sub_roi_y, sub_roi_width, sub_roi_height

def preprocess_image_for_contour_detection(image: np.ndarray, showSteps: bool = False) -> np.ndarray:
    """
    Preprocesses an image for contour detection.
    
    Args:
        image: Input image
        showSteps: Whether to display intermediate steps
        
    Returns:
        Binary thresholded image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply stronger Gaussian blur with larger kernel to reduce noise more effectively
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
    # Apply second blur pass for additional noise reduction
    blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
    
    if showSteps:
        cv2.imshow("After Strong Blur", blurred)
        cv2.waitKey(0)
    
    # Apply stronger CLAHE to enhance contrast more aggressively
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(6, 6))
    enhanced = clahe.apply(blurred)
    
    if showSteps:
        cv2.imshow("After Strong CLAHE Enhancement", enhanced)
        cv2.waitKey(0)
    
    # Apply regular thresholding with a fixed value
    _, thresh = cv2.threshold(enhanced, 125, 255, cv2.THRESH_BINARY_INV)
    
    return thresh

def filter_contours_by_quality(contours: List[np.ndarray], center: Tuple[int, int], 
                              width: int, height: int) -> List[Tuple[np.ndarray, float, float, int, float]]:
    """
    Filters contours by quality metrics (size, centrality, regularity).
    
    Args:
        contours: List of contours
        center: Center point of the image
        width: Width of the image
        height: Height of the image
        
    Returns:
        List of filtered contours with metadata
    """
    central_contours = []
    
    for i, contour in enumerate(contours):
        # Skip small contours
        area = cv2.contourArea(contour)
        if area < 600:
            continue
            
        # Calculate center of contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w//2, y + h//2
            
        # Calculate normalized distance to center (0-1 range)
        distance_to_center = np.sqrt((cx - center[0])**2 + (cy - center[1])**2) / (width/2)
        
        # If too far from center, skip
        if distance_to_center > 0.25:
            continue
        
        # Check if contour touches the image boundaries
        x, y, w, h = cv2.boundingRect(contour)
        border_margin = 2  # Pixels from border to consider as touching
        if x <= border_margin or y <= border_margin or x + w >= width - border_margin or y + h >= height - border_margin:
            # Contour touches or is very close to border, likely not our target
            continue
            
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate circularity to filter out very irregular shapes
        circularity = 0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Filter out extremely irregular shapes
        if circularity < 0.2:
            continue
            
        # Calculate convexity as additional measure of regularity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # Filter out shapes with low convexity
        if convexity < 0.7:
            continue
        
        # Add valid central contour with quality score (combines centrality and size)
        quality_score = area * (1.0 - distance_to_center)
        central_contours.append((contour, area, distance_to_center, i, quality_score))
    
    # Sort central contours by quality score (best first)
    central_contours.sort(key=lambda x: x[4], reverse=True)
    
    return central_contours

def is_contour_inside(inner_contour: np.ndarray, outer_contour: np.ndarray) -> bool:
    """
    Checks if one contour is inside another.
    
    Args:
        inner_contour: The contour that might be inside
        outer_contour: The potential outer contour
        
    Returns:
        True if inner_contour is inside outer_contour, False otherwise
    """
    # More robust check using multiple points of the inner contour
    num_points = len(inner_contour)
    sample_step = max(1, num_points // 10)  # Check ~10 points for efficiency
    
    for i in range(0, num_points, sample_step):
        point = inner_contour[i][0]
        if cv2.pointPolygonTest(outer_contour, (float(point[0]), float(point[1])), False) < 0:
            return False
    return True

def find_primary_and_nested_contours(central_contours: List[Tuple[np.ndarray, float, float, int, float]], 
                                    all_contours: List[np.ndarray],
                                    sub_roi_image: np.ndarray, 
                                    sub_roi_x: int, sub_roi_y: int,
                                    sub_roi_width: int, sub_roi_height: int,
                                    sub_roi_center: Tuple[int, int],
                                    showSteps: bool = False) -> List[np.ndarray]:
    """
    Finds the primary contour and all nested contours within it.
    
    Args:
        central_contours: List of central contours with metadata
        all_contours: List of all detected contours
        sub_roi_image: Sub-ROI image
        sub_roi_x, sub_roi_y: Sub-ROI coordinates
        sub_roi_width, sub_roi_height: Sub-ROI dimensions
        sub_roi_center: Center point of Sub-ROI
        showSteps: Whether to display intermediate steps
        
    Returns:
        List of contours (primary contour first, followed by nested contours)
    """
    filtered_contours = []
    
    if central_contours:
        # Get the best central contour
        largest_contour = central_contours[0][0]
        
        # First add the largest contour
        adjusted_largest = largest_contour.copy()
        adjusted_largest[:, :, 0] += sub_roi_x
        adjusted_largest[:, :, 1] += sub_roi_y
        filtered_contours.append(adjusted_largest)
        
        # Find all contours inside the largest one
        for idx, (contour, area, distance, contour_idx, _) in enumerate(central_contours[1:]):
            largest_area = cv2.contourArea(largest_contour)
            
            # More lenient size threshold for inner contours
            if area < largest_area * 0.02:  # Lowered from 0.05 to detect smaller inner features
                continue
            
            # Check if this contour is inside the largest contour
            if is_contour_inside(contour, largest_contour):
                adjusted_contour = contour.copy()
                adjusted_contour[:, :, 0] += sub_roi_x
                adjusted_contour[:, :, 1] += sub_roi_y
                filtered_contours.append(adjusted_contour)
        
        if showSteps:
            # Visualize largest contour and contours inside it
            main_vis = sub_roi_image.copy()
            # Draw largest contour in red
            cv2.drawContours(main_vis, [largest_contour], -1, (0, 0, 255), 2)
            
            # Draw inner contours in green
            for contour, area, distance, contour_idx, _ in central_contours[1:]:
                largest_area = cv2.contourArea(largest_contour)
                if area >= largest_area * 0.02 and is_contour_inside(contour, largest_contour):
                    cv2.drawContours(main_vis, [contour], -1, (0, 255, 0), 1)
            
            cv2.imshow("Largest Contour and Inner Contours", main_vis)
            cv2.waitKey(0)
    
    # If no valid contours found, try fallback strategies
    if not filtered_contours:
        filtered_contours = find_fallback_contours(
            all_contours, sub_roi_image, sub_roi_x, sub_roi_y,
            sub_roi_width, sub_roi_height, sub_roi_center, showSteps
        )
    
    return filtered_contours

def find_fallback_contours(contours: List[np.ndarray], sub_roi_image: np.ndarray, 
                          sub_roi_x: int, sub_roi_y: int, 
                          sub_roi_width: int, sub_roi_height: int,
                          sub_roi_center: Tuple[int, int],
                          showSteps: bool = False) -> List[np.ndarray]:
    """
    Finds fallback contours when primary contour detection fails.
    
    Args:
        contours: List of all detected contours
        sub_roi_image: Sub-ROI image
        sub_roi_x, sub_roi_y: Sub-ROI coordinates
        sub_roi_width, sub_roi_height: Sub-ROI dimensions
        sub_roi_center: Center point of Sub-ROI
        showSteps: Whether to display intermediate steps
        
    Returns:
        List of fallback contours
    """
    # Try to find the most centered contour that's not near borders
    most_centered_contour = find_most_centered_contour(
        contours, sub_roi_center, sub_roi_width, sub_roi_height
    )
    
    if most_centered_contour is not None:
        # Check if contour is too small relative to sub-ROI
        most_centered_area = cv2.contourArea(most_centered_contour)
        sub_roi_total_area = sub_roi_width * sub_roi_height
        area_ratio = most_centered_area / sub_roi_total_area
        
        if area_ratio < 0.05:  # If contour occupies less than 5% of sub-ROI area
            # Find contours with similar color
            similar_contours = find_contours_with_similar_color(
                most_centered_contour, contours, sub_roi_image, 
                sub_roi_width, sub_roi_height
            )
            
            if len(similar_contours) > 1:  # If we found more similar contours
                # Sort contours by area (largest first)
                similar_contours.sort(key=cv2.contourArea, reverse=True)
                
                # Adjust coordinates of selected contours
                adjusted_contours = []
                for contour in similar_contours:
                    adjusted_contour = contour.copy()
                    adjusted_contour[:, :, 0] += sub_roi_x
                    adjusted_contour[:, :, 1] += sub_roi_y
                    adjusted_contours.append(adjusted_contour)
                
                if showSteps:
                    print(f"Main contour too small. Using {len(similar_contours)} contours with similar color.")
                    group_vis = sub_roi_image.copy()
                    cv2.drawContours(group_vis, [similar_contours[0]], -1, (0, 0, 255), 2)  # Main contour in red
                    for c in similar_contours[1:]:
                        cv2.drawContours(group_vis, [c], -1, (0, 255, 0), 1)  # Secondary contours in green
                    cv2.imshow("Group of similar contours", group_vis)
                    cv2.waitKey(0)
                
                return adjusted_contours
            else:
                # Just use the original centered contour
                adjusted_contour = most_centered_contour.copy()
                adjusted_contour[:, :, 0] += sub_roi_x
                adjusted_contour[:, :, 1] += sub_roi_y
                
                if showSteps:
                    print("Small contour found, but no similar contours nearby.")
                    fallback_image = sub_roi_image.copy()
                    cv2.drawContours(fallback_image, [most_centered_contour], -1, (0, 255, 0), 2)
                    cv2.imshow("Most centered contour", fallback_image)
                    cv2.waitKey(0)
                
                return [adjusted_contour]
        else:
            # Contour is large enough, use it directly
            adjusted_contour = most_centered_contour.copy()
            adjusted_contour[:, :, 0] += sub_roi_x
            adjusted_contour[:, :, 1] += sub_roi_y
            
            if showSteps:
                print("No ideal contours found. Using the most centered contour away from borders.")
                fallback_image = sub_roi_image.copy()
                cv2.drawContours(fallback_image, [most_centered_contour], -1, (0, 255, 0), 2)
                cv2.imshow("Most centered contour", fallback_image)
                cv2.waitKey(0)
            
            return [adjusted_contour]
    
    # Final fallback: create a circular contour
    mask = np.zeros((sub_roi_height, sub_roi_width), dtype=np.uint8)
    radius = min(sub_roi_width, sub_roi_height) // 3
    cv2.circle(mask, (sub_roi_center[0], sub_roi_center[1]), radius, 255, -1)
    circle_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    adjusted_circle_contours = []
    for contour in circle_contours:
        adjusted_contour = contour.copy()
        adjusted_contour[:, :, 0] += sub_roi_x
        adjusted_contour[:, :, 1] += sub_roi_y
        adjusted_circle_contours.append(adjusted_contour)
    
    if showSteps:
        print("No usable contours found. Using circular fallback contour.")
        fallback_image = sub_roi_image.copy()
        cv2.drawContours(fallback_image, circle_contours, -1, (0, 255, 0), 2)
        cv2.imshow("Circular fallback contour", fallback_image)
        cv2.waitKey(0)
    
    return adjusted_circle_contours

def find_most_centered_contour(contours: List[np.ndarray], center: Tuple[int, int], 
                              width: int, height: int) -> Optional[np.ndarray]:
    """
    Finds the most centered contour that's not near image borders.
    
    Args:
        contours: List of contours
        center: Center point coordinates
        width: Image width
        height: Image height
        
    Returns:
        Most centered contour or None if no suitable contour found
    """
    most_centered_contour = None
    min_distance = float('inf')
    border_margin = 150  # Pixels from border to consider as "touching"
    
    for contour in contours:
        # Calculate area to avoid very small contours
        area = cv2.contourArea(contour)
        if area < 200:  # Lower threshold than in main filter
            continue
        
        # Check that contour doesn't touch sub-ROI borders
        x, y, w, h = cv2.boundingRect(contour)
        if (x <= border_margin or y <= border_margin or 
            x + w >= width - border_margin or 
            y + h >= height - border_margin):
            # Contour touches or is very close to border, ignore it
            continue
            
        # Calculate contour center
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w//2, y + h//2
        
        # Calculate distance to center
        distance = np.sqrt((cx - center[0])**2 + (cy - center[1])**2)
        
        if distance < min_distance:
            min_distance = distance
            most_centered_contour = contour
    
    return most_centered_contour

def find_contours_with_similar_color(target_contour: np.ndarray, all_contours: List[np.ndarray], 
                                    image: np.ndarray, width: int, height: int) -> List[np.ndarray]:
    """
    Finds contours with similar color to the target contour.
    
    Args:
        target_contour: Target contour to match color
        all_contours: List of all contours to check
        image: Image containing the contours
        width: Image width
        height: Image height
        
    Returns:
        List of contours with similar color to target
    """
    # Get bounding box of target contour
    x, y, w, h = cv2.boundingRect(target_contour)
    center_x, center_y = x + w//2, y + h//2
    
    # Create mask to extract color of current contour
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [target_contour], -1, 255, -1)
    
    # Get average color inside the contour
    mean_color = cv2.mean(image, mask=mask)[:3]  # BGR
    
    # Find similar-colored contours
    similar_contours = [target_contour]  # Include the target contour itself
    
    for contour in all_contours:
        if np.array_equal(contour, target_contour):
            continue
            
        area = cv2.contourArea(contour)
        if area < 100:  # Ignore very small contours
            continue
            
        # Check proximity
        c_x, c_y, c_w, c_h = cv2.boundingRect(contour)
        c_center_x, c_center_y = c_x + c_w//2, c_y + c_h//2
        
        # Calculate distance between centers
        distance = np.sqrt((center_x - c_center_x)**2 + (center_y - c_center_y)**2)
        max_distance = max(width, height) * 0.3  # 30% of the larger dimension
        
        if distance > max_distance:
            continue
        
        # Compare color
        c_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(c_mask, [contour], -1, 255, -1)
        c_mean_color = cv2.mean(image, mask=c_mask)[:3]
        
        # Calculate color difference (Euclidean distance in RGB space)
        color_diff = np.sqrt(np.sum((np.array(mean_color) - np.array(c_mean_color))**2))
        color_threshold = 30  # Color difference threshold
        
        if color_diff < color_threshold:
            similar_contours.append(contour)
    
    return similar_contours

if __name__ == "__main__":
    image_path = "your_image_path_here.jpg"  # Replace with your image path if you want to test
    print(f"Processing image: {image_path}")
    
    # Load the image directly
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    
    # Segment the image to find the 3D figure
    segmented_image, centered_contour = segment_3d_figure(original_image, showSteps=True)
    
    if centered_contour is None:
        print("Failed to find 3D figure in the image")
    else:
        print(f"Found {len(centered_contour)} points in the centered contour")