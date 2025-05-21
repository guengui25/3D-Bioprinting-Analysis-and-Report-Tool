import cv2
import numpy as np
import os
from skimage import morphology
from typing import Dict, List, Tuple, Optional, Any
from typing import Tuple, List

from functions.width_detection.metrics_exporter import export_all_metrics

# Constants for analysis parameters
MAX_SAMPLE_DIST = 15  # Maximum sampling distance when measuring line width
DISTANCE_CONTACT_MULTIPLIER = 1.2  # Multiplier to determine if a line touches a contour
MIN_SKELETON_POINTS = 3  # Minimum number of points required for reliable skeleton analysis

def normalize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Normalize a binary mask to uint8 with values 0 or 255.
    
    Args:
        mask: Input binary mask array
        
    Returns:
        Normalized mask with values 0 or 255
    """
    mask = np.array(mask, dtype=np.uint8)
    return mask * 255 if mask.max() <= 1 else mask

def compute_distance_transform(mask: np.ndarray) -> np.ndarray:
    """
    Compute distance transform on the inverse of a contour mask.
    
    Args:
        mask: Binary mask image
        
    Returns:
        Distance transform array where each pixel value represents 
        the distance to the nearest zero pixel
    """
    inv_mask = cv2.bitwise_not(mask)
    return cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)

def create_contour_masks(external_contour: np.ndarray, internal_contours: List[np.ndarray], 
                         height: int, width: int) -> Dict[str, np.ndarray]:
    """
    Creates masks for external and internal contours.
    
    Args:
        external_contour: Main contour of the figure
        internal_contours: List of internal contours (holes)
        height, width: Image dimensions
        
    Returns:
        Dictionary with different contour masks
    """
    # Initialize empty masks
    figure_mask = np.zeros((height, width), dtype=np.uint8)
    external_contour_mask = np.zeros((height, width), dtype=np.uint8)
    internal_areas_mask = np.zeros((height, width), dtype=np.uint8)
    internal_contours_mask = np.zeros((height, width), dtype=np.uint8)
    all_contours_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create complete figure mask and its contour
    cv2.drawContours(figure_mask, [external_contour], -1, 255, -1)  # Fill the entire figure
    cv2.drawContours(external_contour_mask, [external_contour], -1, 255, 1)  # Draw only border
    cv2.drawContours(all_contours_mask, [external_contour], -1, 255, 1)  # Add external to combined contours
    
    # Process each internal contour (holes in the figure)
    for contour in internal_contours:
        temp_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(temp_mask, [contour], -1, 255, -1)  # Fill the internal contour
        internal_areas_mask = cv2.bitwise_or(internal_areas_mask, temp_mask)  # Combine with previous internal areas
        
        cv2.drawContours(internal_contours_mask, [contour], -1, 255, 1)  # Draw only border
        cv2.drawContours(all_contours_mask, [contour], -1, 255, 1)  # Add to combined contours
    
    # Create valid area (figure without internal holes)
    valid_area = cv2.bitwise_and(figure_mask, cv2.bitwise_not(internal_areas_mask))
    
    return {
        "figure_mask": figure_mask,
        "external_contour_mask": external_contour_mask,
        "internal_areas_mask": internal_areas_mask,
        "internal_contours_mask": internal_contours_mask,
        "all_contours_mask": all_contours_mask,
        "valid_area": valid_area
    }

def calculate_line_direction(line_mask: np.ndarray) -> Dict[str, Any]:
    """
    Calculates the main direction of the line using PCA.
    
    Args:
        line_mask: Binary mask of the line
        
    Returns:
        Dictionary containing orientation information including:
        - is_horizontal: Boolean indicating if line is more horizontal than vertical
        - main_direction: Principal direction vector
        - perpendicular_direction: Vector perpendicular to main direction
        - skeleton_points: Array of points along the line skeleton
    """
    # Create skeleton from line mask to get centerline points
    skeleton = morphology.skeletonize(line_mask > 0)
    skeleton_points = np.column_stack(np.where(skeleton > 0))
    
    # Handle empty skeleton case
    if len(skeleton_points) == 0:
        return {
            "is_horizontal": True,
            "main_direction": np.array([1, 0]),
            "perpendicular_direction": np.array([0, 1]),
            "skeleton_points": np.array([])
        }
    
    try:
        # Use PCA (Principal Component Analysis) to find main direction
        covariance_matrix = np.cov(skeleton_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # The eigenvector with largest eigenvalue represents the main direction
        main_direction = eigenvectors[:, np.argmax(eigenvalues)]
        main_direction = main_direction / np.linalg.norm(main_direction)
        
        # Calculate perpendicular vector (90 degrees rotation)
        perpendicular_direction = np.array([-main_direction[1], main_direction[0]])
        perpendicular_direction = perpendicular_direction / np.linalg.norm(perpendicular_direction)
        
        # Determine if the line is more horizontal or vertical
        is_horizontal = abs(main_direction[1]) < abs(main_direction[0])
    except:
        # Fallback to default values if PCA fails
        is_horizontal = True
        main_direction = np.array([1, 0])
        perpendicular_direction = np.array([0, 1])
    
    return {
        "is_horizontal": is_horizontal,
        "main_direction": main_direction,
        "perpendicular_direction": perpendicular_direction,
        "skeleton_points": skeleton_points
    }

def measure_line_width(skeleton_points: np.ndarray, perpendicular_direction: np.ndarray, 
                       valid_area: np.ndarray, all_contours_mask: np.ndarray, 
                       height: int, width: int) -> Tuple[int, List[int]]:
    """
    Measures the typical width of the line based on distance to contours.
    
    Args:
        skeleton_points: Array of points along the line skeleton
        perpendicular_direction: Direction vector perpendicular to the main line direction
        valid_area: Mask of the valid areas in the figure
        all_contours_mask: Combined mask of all contours
        height, width: Image dimensions
        
    Returns:
        Typical width of the line in pixels and list of all width measurements
    """
    if len(skeleton_points) == 0:
        return 5, []  # Default value if no skeleton points available
    
    # Store width measurements from multiple sample points
    width_samples = []
    max_sample_dist = min(MAX_SAMPLE_DIST, min(height, width) // 12)
    
    sample_step = max(1, len(skeleton_points) // 100)  # Very fine sampling for accuracy

    # Iterate through sample points along the skeleton
    for idx in range(0, len(skeleton_points), sample_step):
        y, x = skeleton_points[idx]

        # Measure distance in positive perpendicular direction
        dy, dx = perpendicular_direction
        dist_pos = 0
        for d in range(1, max_sample_dist + 1):
            ny, nx = int(y + d * dy), int(x + d * dx)
            # Stop when reaching image boundary, invalid area, or contour
            if ny < 0 or ny >= height or nx < 0 or nx >= width \
               or valid_area[ny, nx] == 0 or all_contours_mask[ny, nx] > 0:
                dist_pos = d
                break

        # Measure distance in negative perpendicular direction
        dy_neg, dx_neg = -dy, -dx
        dist_neg = 0
        for d in range(1, max_sample_dist + 1):
            ny, nx = int(y + d * dy_neg), int(x + d * dx_neg)
            # Stop when reaching image boundary, invalid area, or contour
            if ny < 0 or ny >= height or nx < 0 or nx >= width \
               or valid_area[ny, nx] == 0 or all_contours_mask[ny, nx] > 0:
                dist_neg = d
                break

        # Total width is the sum of distances in both directions
        width_samples.append(dist_pos + dist_neg)

    # Use median to avoid influence of outliers
    typical_width = int(np.median(width_samples)) if width_samples else max_sample_dist
    return typical_width, width_samples

def detect_contour_contact(skeleton_points: np.ndarray, external_contour_mask: np.ndarray, 
                           internal_contours_mask: np.ndarray, typical_width: float) -> Tuple[bool, bool]:
    """
    Detects if the line touches or is close to touching external or internal contours.
    
    Args:
        skeleton_points: Array of points along the line skeleton
        external_contour_mask: Mask of the external contour
        internal_contours_mask: Mask of all internal contours
        typical_width: Typical width of the line in pixels
        
    Returns:
        Tuple of (touches_external, touches_internal) booleans
    """
    if len(skeleton_points) == 0:
        return False, False
    
    # Calculate distance transform for both contour types
    dist_ext = compute_distance_transform(external_contour_mask)
    dist_int = compute_distance_transform(internal_contours_mask)

    # Extract distance values at skeleton points
    external_distances = dist_ext[skeleton_points[:, 0], skeleton_points[:, 1]]
    internal_distances = dist_int[skeleton_points[:, 0], skeleton_points[:, 1]]

    # Use median distance to determine proximity to contours
    median_ext = np.median(external_distances) if external_distances.size > 0 else np.inf
    median_int = np.median(internal_distances) if internal_distances.size > 0 else np.inf

    # Line is considered "touching" if median distance is less than threshold
    # The threshold is a multiple of the typical line width
    threshold = typical_width * DISTANCE_CONTACT_MULTIPLIER

    touches_external = median_ext < threshold
    touches_internal = median_int < threshold
    
    return touches_external, touches_internal

def expand_line_to_contours(line_mask: np.ndarray, mask_info: Dict[str, np.ndarray], 
                            direction_info: Dict[str, Any], touches_external: bool, 
                            touches_internal: bool, typical_width: float) -> np.ndarray:
    """
    Expands line mask to relevant contours, with asymmetric expansion adapting to closest contours.
    
    Args:
        line_mask: Binary mask of the line
        mask_info: Dictionary containing various contour masks
        direction_info: Dictionary with line direction information
        touches_external: Whether the line touches external contours
        touches_internal: Whether the line touches internal contours
        typical_width: Typical width of the line in pixels
        
    Returns:
        Expanded binary mask of the line area
    """
    # Ensure binary mask has proper format
    line_mask = normalize_mask(line_mask)
    
    # Extract masks from dictionary
    valid_area = mask_info["valid_area"]
    figure_mask = mask_info["figure_mask"]
    all_contours_mask = mask_info["all_contours_mask"]
    external_contour_mask = mask_info["external_contour_mask"]
    internal_contours_mask = mask_info["internal_contours_mask"]
    
    # Extract direction information
    skeleton_points = direction_info["skeleton_points"]
    perpendicular_direction = direction_info["perpendicular_direction"]
    height, width = valid_area.shape[:2]
    
    # Simple dilation for very thin lines with few skeleton points
    if len(skeleton_points) < MIN_SKELETON_POINTS:
        expanded_mask = cv2.dilate(line_mask, np.ones((5, 5), np.uint8), iterations=1)
        return cv2.bitwise_and(expanded_mask, valid_area)
    
    # Calculate average widths in each direction of the line
    positive_side_widths = []
    negative_side_widths = []
    
    # Sample points along the skeleton to determine width distribution
    for i in range(0, len(skeleton_points), max(1, len(skeleton_points) // 20)):
        y, x = skeleton_points[i]
        
        # Measure width toward positive perpendicular direction
        for dist in range(1, 40):
            ny = int(y + dist * perpendicular_direction[0])
            nx = int(x + dist * perpendicular_direction[1])
            if not (0 <= ny < height and 0 <= nx < width) or valid_area[ny, nx] == 0 or all_contours_mask[ny, nx] > 0:
                positive_side_widths.append(dist)
                break
                
        # Measure width toward negative perpendicular direction
        for dist in range(1, 40):
            ny = int(y - dist * perpendicular_direction[0])
            nx = int(x - dist * perpendicular_direction[1])
            if not (0 <= ny < height and 0 <= nx < width) or valid_area[ny, nx] == 0 or all_contours_mask[ny, nx] > 0:
                negative_side_widths.append(dist)
                break
    
    # Calculate median width for each side (more robust than mean)
    average_pos_width = np.median(positive_side_widths) if positive_side_widths else typical_width/2
    average_neg_width = np.median(negative_side_widths) if negative_side_widths else typical_width/2
    
    # Initialize expansion mask with skeleton
    expansion_mask = np.zeros_like(line_mask)
    for y, x in skeleton_points:
        expansion_mask[y, x] = 255
    
    # Optimize processing by using a subset of points
    sampling_step = max(1, len(skeleton_points) // 40)
    
    # Expand from each sampled skeleton point
    for i in range(0, len(skeleton_points), sampling_step):
        y, x = skeleton_points[i]
        
        # For each perpendicular direction (positive and negative)
        for direction in [1, -1]:
            dy = direction * perpendicular_direction[0]
            dx = direction * perpendicular_direction[1]
            
            # Determine which average width to use based on direction
            average_width = average_pos_width if direction == 1 else average_neg_width
            
            # Scan to detect nearby contours (including image border)
            found_contour = False
            contour_dist = -1
            contour_type = None
            
            # Look for contours or boundaries in this direction
            for dist in range(1, 50):  # Wide distance to detect any contour
                ny, nx = int(y + dist * dy), int(x + dist * dx)
                
                # Check if we leave the image or valid area
                if not (0 <= ny < height and 0 <= nx < width) or valid_area[ny, nx] == 0:
                    found_contour = True
                    contour_dist = dist
                    contour_type = "border"
                    break
                
                # Check if we find external contour
                if external_contour_mask[ny, nx] > 0:
                    found_contour = True
                    contour_dist = dist
                    contour_type = "external"
                    break
                    
                # Check if we find internal contour
                if internal_contours_mask[ny, nx] > 0:
                    found_contour = True
                    contour_dist = dist
                    contour_type = "internal"
                    break
            
            # Determine expansion distance based on contour type
            if found_contour:
                if contour_type == "external":
                    # For external contours, reach EXACTLY to the contour
                    max_dist = contour_dist
                elif contour_type == "internal":
                    # For internal contours, reach EXACTLY to the contour
                    max_dist = contour_dist
                else:  # "border" or others
                    # For borders, use average width or distance to border, whichever is less
                    max_dist = min(int(average_width), contour_dist - 1)
            else:
                # If no contour found, use average width for this side
                max_dist = int(average_width)
            
            # Mark points from skeleton to calculated distance
            for dist in range(1, max_dist + 1):
                ny, nx = int(y + dist * dy), int(x + dist * dx)
                
                # Check bounds
                if not (0 <= ny < height and 0 <= nx < width) or valid_area[ny, nx] == 0:
                    break
                
                # Mark this point as part of expansion
                expansion_mask[ny, nx] = 255
    
    # Process expanded mask to create a filled shape
    expanded_mask = cv2.dilate(expansion_mask, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(expanded_mask)
    cv2.drawContours(filled_mask, contours, -1, 255, -1)  # Fill contours
    
    # Special handling when line touches external contour
    if touches_external:
        # Slightly dilate external contour to ensure we detect all contacts
        dilated_ext_contour = cv2.dilate(external_contour_mask, np.ones((2, 2), np.uint8), iterations=1)
        
        # Find points where expansion touches external contour
        contact_with_external = cv2.bitwise_and(expanded_mask, dilated_ext_contour)
        
        if np.any(contact_with_external):
            # Define the area where flood fill can occur - between the line and border
            area_between_line_and_border = cv2.bitwise_and(
                figure_mask, 
                cv2.bitwise_not(filled_mask)
            )
            
            # Prepare seed points for flood fill just outside the contour contact
            seeds = cv2.dilate(contact_with_external, np.ones((3, 3), np.uint8), iterations=1)
            seeds = cv2.bitwise_and(seeds, area_between_line_and_border)
            
            if np.any(seeds):
                # Perform flood fill from these seed points
                flooded_mask = np.zeros_like(filled_mask)
                seed_points = np.column_stack(np.where(seeds > 0))
                
                # Limit number of seed points to optimize processing
                max_points = min(20, len(seed_points))
                if max_points > 0:
                    indices = np.linspace(0, len(seed_points)-1, max_points, dtype=int)
                    
                    for idx in indices:
                        y, x = seed_points[idx]
                        cv2.floodFill(
                            flooded_mask, None, (x, y), 255,
                            loDiff=0, upDiff=0,
                            flags=8 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE
                        )
                    
                    # Add flood fill results to the filled mask, staying within valid area
                    filled_mask = cv2.bitwise_or(
                        filled_mask, 
                        cv2.bitwise_and(flooded_mask, valid_area)
                    )
    
    # Finalize the mask
    filled_mask = cv2.bitwise_or(filled_mask, line_mask)  # Ensure original line is included
    final_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # Close small gaps
    final_mask = cv2.bitwise_and(final_mask, valid_area)  # Ensure mask stays within valid area
    
    # If result is too small, use simple dilation as fallback
    if np.sum(final_mask > 0) < 50:
        final_mask = cv2.dilate(line_mask, np.ones((5, 5), np.uint8), iterations=1)
        final_mask = cv2.bitwise_and(final_mask, valid_area)

    return final_mask

def determine_gcode_line_area(segmented_image: np.ndarray, line_mask: np.ndarray, 
                              external_contour: np.ndarray, internal_contours: List[np.ndarray],
                              ) -> Dict[str, Any]:
    """
    Main function that determines the area corresponding to a GCode line.
    
    Args:
        segmented_image: Segmented image of the figure
        line_mask: Binary mask of the G-code line
        external_contour: Main contour of the figure
        internal_contours: List of internal contours (holes)
        
    Returns:
        Dictionary with analysis results including success status, area mask, and metrics
    """
    # Input validation
    if segmented_image is None or line_mask is None:
        return {"success": False, "error": "Invalid inputs"}
    
    # Normalize line mask
    line_mask = normalize_mask(line_mask)
    
    # Create masks for contour analysis
    height, width = segmented_image.shape[:2]
    mask_info = create_contour_masks(external_contour, internal_contours, height, width)
    
    # Check if line intersects with valid area
    line_in_valid_area = cv2.bitwise_and(line_mask, mask_info["valid_area"])
    if np.sum(line_in_valid_area) == 0:
        return {"success": False, "error": "Line does not intersect with valid area"}
    
    # Calculate line direction and skeleton
    direction_info = calculate_line_direction(line_mask)
    if len(direction_info["skeleton_points"]) == 0:
        return {"success": False, "error": "Could not calculate line skeleton"}
    
    # Measure typical width of the line
    typical_width, width_samples = measure_line_width(
        direction_info["skeleton_points"],
        direction_info["perpendicular_direction"],
        mask_info["valid_area"],
        mask_info["all_contours_mask"],
        height, width,
    )
    
    # Detect if line touches contours
    touches_external, touches_internal = detect_contour_contact(
        direction_info["skeleton_points"],
        mask_info["external_contour_mask"],
        mask_info["internal_contours_mask"],
        typical_width
    )
    
    # Expand line to create area mask
    area_mask = expand_line_to_contours(
        line_mask,
        mask_info,
        direction_info,
        touches_external,
        touches_internal,
        typical_width
    )
    
    return {
        "success": True,
        "area_mask": area_mask,
        "typical_width": typical_width,
        "width_samples": width_samples,
        "touches_external": touches_external,
        "touches_internal": touches_internal,
        "is_horizontal": direction_info["is_horizontal"]
    }

def measure_actual_line_width(binary_mask: np.ndarray, transformed_line: Dict[str, Any], scale_factor: Optional[float] = None) -> Dict[str, Any]:
    """
    Measures the actual width of the line and compares it with the expected width.
    
    Args:
        binary_mask: Binary mask of the line area
        transformed_line: Dictionary with information about the expected line
        scale_factor: Pixels-to-mm scale factor (optional)
        
    Returns:
        Dictionary with width measurements including pixel values, 
        real-world units, expected widths, and error metrics
    """
    # Handle invalid input
    if binary_mask is None or binary_mask.size == 0:
        return {"width_pixels": 0, "width_mm": 0, "expected_width_pixels": 0}
    
    # Ensure mask is binary
    binary_mask = binary_mask > 0
    
    # Compute perpendicular sampling for each skeleton point
    # Get direction info (skeleton and perpendicular direction)
    direction_info = calculate_line_direction((binary_mask > 0).astype(np.uint8))
    skeleton_points = direction_info["skeleton_points"]
    perpendicular_direction = direction_info["perpendicular_direction"]

    width_samples = []
    height, width = binary_mask.shape
    max_search = max(height, width)

    dy, dx = perpendicular_direction
    # Sample width at each skeleton point
    for y, x in skeleton_points:
        # Positive direction
        dist_pos = 0
        for d in range(1, max_search):
            ny, nx = int(y + d * dy), int(x + d * dx)
            if ny < 0 or ny >= height or nx < 0 or nx >= width or not binary_mask[ny, nx]:
                dist_pos = d
                break
        # Negative direction
        dist_neg = 0
        for d in range(1, max_search):
            ny, nx = int(y - d * dy), int(x - d * dx)
            if ny < 0 or ny >= height or nx < 0 or nx >= width or not binary_mask[ny, nx]:
                dist_neg = d
                break
        width_samples.append(dist_pos + dist_neg)

    # Compute metrics
    width_pixels = float(np.mean(width_samples)) if width_samples else 0.0
    expected_width_pixels = transformed_line.get('transformed_thickness', 0)
    std_width_pixels = float(np.std(width_samples)) if width_samples else 0.0
    max_width_pixels = float(np.max(width_samples)) if width_samples else 0.0
    min_width_pixels = float(np.min(width_samples)) if width_samples else 0.0
    width_variation = (std_width_pixels / width_pixels * 100) if width_pixels > 0 else 0.0

    # Convert to real world units if scale_factor is provided
    width_mm = width_pixels / scale_factor if scale_factor and scale_factor > 0 else 0.0
    expected_width_mm = expected_width_pixels / scale_factor if scale_factor and scale_factor > 0 else 0.0
    std_width_mm = std_width_pixels / scale_factor if scale_factor and scale_factor > 0 else 0.0
    max_width_mm = max_width_pixels / scale_factor if scale_factor and scale_factor > 0 else 0.0
    min_width_mm = min_width_pixels / scale_factor if scale_factor and scale_factor > 0 else 0.0

    return {
        "width_pixels": width_pixels,
        "width_mm": width_mm,
        "expected_width_pixels": expected_width_pixels,
        "expected_width_mm": expected_width_mm,
        "std_width_pixels": std_width_pixels,
        "std_width_mm": std_width_mm,
        "width_variation_percentage": width_variation,
        "width_max_pixels": max_width_pixels,
        "width_min_pixels": min_width_pixels,
        "width_max_mm": max_width_mm,
        "width_min_mm": min_width_mm,
        "width_samples": width_samples
    }

def analyze_gcode_line(segmented_image: np.ndarray, line_data: Dict[str, Any], 
                       external_contour: np.ndarray, internal_contours: List[np.ndarray], 
                       scale_factor: Optional[float] = None, output_dir: Optional[str] = None, 
                       line_name: str = "", save_images: bool = False) -> Dict[str, Any]:
    """
    Analyzes a G-code line by comparing it with the actual figure.
    
    Args:
        segmented_image: Segmented image of the figure
        line_data: Dictionary with information about the G-code line
        external_contour: Main contour of the figure
        internal_contours: List of internal contours (holes)
        scale_factor: Pixels-to-mm scale factor (optional)
        output_dir: Directory to save output files (optional)
        line_name: Name identifier for the line
        save_images: Whether to save visualization images
        
    Returns:
        Dictionary with analysis results including success status and measurements
    """
    try:
        # Verify input data contains required information
        if "line_mask" not in line_data:
            return {"success": False, "error": "Line mask not found"}
            
        line_mask = normalize_mask(line_data["line_mask"])
        
        if not line_data.get("transformed_lines", []):
            return {"success": False, "error": "No transformed line data"}
        
        # Create contour masks
        height, width = segmented_image.shape[:2]
        mask_info = create_contour_masks(external_contour, internal_contours, height, width)
        valid_area = mask_info["valid_area"]
        
        # Check if line crosses contours (for visualization)
        line_in_valid_area = cv2.bitwise_and(line_mask, valid_area)
        crosses_contours = np.sum(line_mask) > np.sum(line_in_valid_area)
        
        # Determine line area
        area_result = determine_gcode_line_area(
            segmented_image, line_mask, external_contour, internal_contours
        )
        
        if not area_result.get("success", False):
            return {"success": False, "error": area_result.get("error", "Error in area analysis")}
        
        # Measure actual line width and compare with expected width
        width_stats = measure_actual_line_width(area_result["area_mask"], line_data["transformed_lines"][0], scale_factor)
        
        # Generate and save visualization if requested
        if save_images and output_dir:
            generate_visualization(
                segmented_image, line_mask, area_result["area_mask"], 
                output_dir, line_name, crosses_contours
            )
        
        # Return comprehensive analysis results
        return {
            "success": True,
            "line_name": line_name,
            "width_analysis": width_stats,
            "area_mask": area_result["area_mask"],
            "touches_external": area_result["touches_external"],
            "touches_internal": area_result["touches_internal"],
            "crosses_contours": crosses_contours,
            "is_horizontal": area_result["is_horizontal"],
            "typical_width": area_result["typical_width"],
            "width_samples": area_result.get("width_samples", []),
        }
        
    except Exception as e:
        print(f"Unexpected error in analyze_gcode_line: {e}")
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

def generate_visualization(segmented_image: np.ndarray, line_mask: np.ndarray, 
                           area_mask: np.ndarray, output_dir: str, 
                           line_name: str, crosses_contours: bool = False) -> None:
    """
    Generates and saves visualization of line analysis.
    
    Args:
        segmented_image: Segmented image of the figure
        line_mask: Binary mask of the line
        area_mask: Expanded mask of the line area
        output_dir: Directory to save output images
        line_name: Name identifier for the line
        crosses_contours: Whether the line crosses contours
        
    Returns:
        None
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a copy of the image for visualization
        visualization = segmented_image.copy()
        
        # Highlight the original line path
        mask_indices = np.where(line_mask > 0)
        if mask_indices[0].size > 0:
            # Use red for lines that cross contours, blue for normal lines
            if crosses_contours:
                visualization[mask_indices] = (0, 0, 255)  # Red (BGR format)
            else:
                visualization[mask_indices] = (255, 0, 0)  # Blue (BGR format)
        
        # Create semi-transparent overlay for the expanded area
        mask_overlay = np.zeros_like(segmented_image)
        area_indices = np.where(area_mask > 0)
        if area_indices[0].size > 0:
            mask_overlay[area_indices[0], area_indices[1], 1] = 255  # Green channel
        
        # Blend the overlay with the visualization
        cv2.addWeighted(mask_overlay, 0.3, visualization, 0.7, 0, visualization)
        
        # Save the visualization image
        cv2.imwrite(os.path.join(output_dir, f"analysis_{line_name}.png"), visualization)
    except Exception as e:
        print(f"Error saving visualization: {e}")

def prepare_contour_data(overlay_results: Dict[str, Any]) -> Tuple[np.ndarray, List[np.ndarray], Optional[float]]:
    """
    Prepares contour data for analysis.
    
    Args:
        overlay_results: Dictionary with overlay analysis results
        
    Returns:
        Tuple containing:
        - External contour array
        - List of internal contour arrays
        - Scale factor (pixels to mm)
    """
    try:
        # Extract external contour
        external_contour = overlay_results.get("figure_contour")
        
        # Convert list to numpy array if needed
        if isinstance(external_contour, list):
            external_contour = np.array(external_contour, dtype=np.int32)
        
        # Extract and process internal contours
        internal_contours = overlay_results.get("internal_contours", [])
        internal_contours_np = []
        
        if internal_contours:
            print(f"Using {len(internal_contours)} internal contours")
            # Convert each internal contour to numpy array
            for cnt in internal_contours:
                if isinstance(cnt, list):
                    cnt = np.array(cnt, dtype=np.int32)
                internal_contours_np.append(cnt)
        else:
            print("No internal contours found")
        
        # Extract scale factor (pixels to mm)
        scale_factor = overlay_results.get("scale_info", {}).get("image_scale")
        
        return external_contour, internal_contours_np, scale_factor
        
    except Exception as e:
        print(f"Error processing contours: {e}")
        return None, [], None

def process_gcode_analysis(overlay_results: Dict[str, Any], segmented_image: np.ndarray, 
                           output_dir: str, save_images: bool = True,
                           export_samples_csv: bool = True, num_samples: int = 20) -> Dict[str, Any]:
    """
    Processes all G-code lines for analysis.
    
    Args:
        overlay_results: Dictionary with overlay analysis results
        segmented_image: Segmented image of the figure
        output_dir: Directory to save output files
        save_images: Whether to save visualization images
        export_samples_csv: Whether to export per-sample width measurements
        
    Returns:
        Dictionary with analysis results for all lines and summary metrics
    """    
    # Verify overlay results are valid
    if not overlay_results.get("success", False):
        return {"success": False, "error": "Invalid overlay results"}
    
    # Create analysis directory
    analysis_dir = os.path.join(output_dir, "line_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Create images directory if saving images
    images_dir = os.path.join(analysis_dir, "images") if save_images else None
    if save_images:
        os.makedirs(images_dir, exist_ok=True)
    
    # Prepare contour data for analysis
    external_contour, internal_contours_np, scale_factor = prepare_contour_data(overlay_results)
    
    # Process each line
    all_line_results = []
    analyzed_lines_count = 0
    
    for line_data in overlay_results.get("lines", []):
        # Skip lines that failed in previous processing
        if not line_data.get("success", False):
            continue
        
        # Get line name or generate default
        line_name = line_data.get("line_name", f"line_{len(all_line_results)}")
        print(f"Analyzing line: {line_name}")
        
        # Analyze this line
        line_result = analyze_gcode_line(
            segmented_image, line_data, external_contour, internal_contours_np, scale_factor,
            output_dir=images_dir, line_name=line_name, save_images=save_images
        )
        
        # Store successful results
        if line_result.get("success", False):
            all_line_results.append(line_result)
            analyzed_lines_count += 1
        else:
            print(f"Error analyzing line {line_name}: {line_result.get('error', 'unknown error')}")
    
    # Export all metrics to CSV and generate statistics
    metrics_result = export_all_metrics(
        all_line_results, analysis_dir,
        num_samples=num_samples,
        scale_factor=scale_factor,
        export_samples_csv=export_samples_csv
    )
    
    # Log completion information
    print(f"Analysis completed. Results saved in: {analysis_dir}")
    print(f"{analyzed_lines_count} lines successfully analyzed")
    
    # Return comprehensive results
    return {
        "success": True,
        "analyzed_lines": len(all_line_results),
        "summary": metrics_result.get("summary", {}),
        "line_results": all_line_results,
        "output_dir": analysis_dir,
        "csv_path": metrics_result.get("csv_path", "")
    }