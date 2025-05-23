import cv2
import numpy as np
import os
import glob
from typing import List, Dict, Tuple, Optional, Any

from functions.overlay_gcode.scale_gcode import extract_scale_from_image

def extract_green_line(input_image_path: str, output_dir: Optional[str] = None, 
                      showSteps: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
    """
    Extracts the green line from a G-code image generated by plot_timelapse_highlight_current.
    
    Args:
        input_image_path: Path to the input image
        output_dir: Directory to save results
        showSteps: Show intermediate steps
        
    Returns:
        Tuple containing green mask, detected lines, and original image
    """
    # Read image
    image = cv2.imread(input_image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_image_path}")
    
    # Extract only the green color (#00FF00) with permissive thresholds
    b, g, r = cv2.split(image)
    
    # Create mask with permissive thresholds to capture all green variations
    green_mask = (g > 150) & (b < 100) & (r < 100)
    green_mask = green_mask.astype(np.uint8) * 255
    
    # Find contours to extract lines
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract line features (polygonal approximation)
    lines = []
    for contour in contours:
        # Ignore very small contours (might be noise)
        if cv2.contourArea(contour) < 2:
            continue
            
        # Extract key points of the contour (polygonal approximation)
        epsilon = 0.5  # Precision of approximation
        approx = cv2.approxPolyDP(contour, epsilon, closed=False)
        
        # Calculate average thickness (estimation based on area/length)
        if len(approx) > 1:
            # Contour length
            perimeter = cv2.arcLength(contour, closed=False)
            # Contour area
            area = cv2.contourArea(contour)
            # Thickness estimation (area / length)
            thickness = max(1, int(area / max(1, perimeter) * 2))
        else:
            thickness = 1
        
        # Save line information
        lines.append({
            'contour': contour,
            'approx': approx,
            'thickness': thickness,
            'color': (0, 255, 0)  # Green color
        })
    
    if showSteps:
        # Show original image and mask
        cv2.imshow('Original image', image)
        cv2.imshow('Green mask', green_mask)
        
        # Show detected lines
        line_preview = image.copy()
        for line in lines:
            cv2.drawContours(line_preview, [line['contour']], -1, (0, 255, 255), 1)
            # Draw key points
            for point in line['approx']:
                cv2.circle(line_preview, tuple(point[0]), 2, (255, 0, 255), -1)
        cv2.imshow('Detected lines', line_preview)
        cv2.waitKey(500)
    
    return green_mask, lines, image

def draw_transformed_lines(lines: List[Dict[str, Any]], transform_matrix: np.ndarray, 
                          target_image: np.ndarray, contour: Optional[np.ndarray] = None, 
                          showSteps: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms detected lines and draws them on the target image.
    
    Args:
        lines: List of detected lines
        transform_matrix: Transformation matrix
        target_image: Image to draw lines on
        contour: Contour to draw on the image (optional)
        showSteps: Whether to show intermediate steps
        
    Returns:
        Image with drawn lines and mask of the lines
    """
    h, w = target_image.shape[:2]
    
    # Create result image
    result = target_image.copy()
    
    # Draw contour if it exists
    if contour is not None:
        cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)
    
    # Create mask for lines
    line_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Process each line
    for line in lines:
        # Transform contour points
        contour_points = line['contour'].reshape(-1, 2).astype(np.float32)
        transformed_points = cv2.transform(contour_points.reshape(-1, 1, 2), transform_matrix)
        transformed_contour = transformed_points.reshape(-1, 1, 2).astype(np.int32)
        
        # Transform key points (polygonal approximation)
        approx_points = line['approx'].reshape(-1, 2).astype(np.float32)
        transformed_approx = cv2.transform(approx_points.reshape(-1, 1, 2), transform_matrix)
        transformed_approx = transformed_approx.reshape(-1, 1, 2).astype(np.int32)
        
        # Draw line in the mask - Calculate thickness proportionally
        thickness = max(1, int(line['thickness'] * transform_matrix[0, 0]))  # Scale thickness
        
        # If the line has at least 2 points, draw segments between them
        if len(transformed_approx) >= 2:
            # Draw segments between key points more precisely
            for i in range(len(transformed_approx) - 1):
                pt1 = tuple(transformed_approx[i][0])
                pt2 = tuple(transformed_approx[i + 1][0])
                
                # Verify points are within image bounds
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(result, pt1, pt2, line['color'], thickness)
                    cv2.line(line_mask, pt1, pt2, 255, thickness)
        else:
            # If there's only one point or insufficient points, use the complete contour
            # This helps maintain details in complex lines
            for i in range(len(transformed_contour)-1):
                if i >= len(transformed_contour) - 1:
                    continue
                    
                pt1 = tuple(transformed_contour[i][0])
                pt2 = tuple(transformed_contour[i+1][0])
                
                # Verify points are within image bounds
                if (0 <= pt1[0] < w and 0 <= pt1[1] < h and 
                    0 <= pt2[0] < w and 0 <= pt2[1] < h):
                    cv2.line(result, pt1, pt2, line['color'], thickness)
                    cv2.line(line_mask, pt1, pt2, 255, thickness)
    
    if showSteps:
        cv2.imshow('Transformed lines', result)
        cv2.imshow('Line mask', line_mask)
        cv2.waitKey(500)
    
    return result, line_mask

def calculate_global_transformation(first_image: np.ndarray, contour_image: np.ndarray, 
                                   scale_factor: Optional[float] = None, 
                                   gcode_scale_factor: Optional[float] = None,
                                   showSteps: bool = False) -> Optional[np.ndarray]:
    """
    Calculates global transformation matrix between G-code image and target image.
    
    Args:
        first_image: First G-code image
        contour_image: Image with contour
        scale_factor: Scale factor in pixels per unit (optional)
        gcode_scale_factor: G-code scale factor (optional)
        showSteps: Whether to show intermediate steps
        
    Returns:
        Transformation matrix or None if calculation fails
    """
    try:
        # Calculate real scale between G-code and target image
        if scale_factor is not None and gcode_scale_factor is not None:
            real_scale = scale_factor / gcode_scale_factor
            print(f"Calculated real scale: image={scale_factor} px/mm, gcode={gcode_scale_factor} px/mm, factor={real_scale:.4f}")
        else:
            real_scale = 1.0
            print("No scale applied (factors not available)")
        
        # Convert to grayscale to find contours
        gray_gcode = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        gray_contour = cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find contours
        _, thresh_gcode = cv2.threshold(gray_gcode, 127, 255, cv2.THRESH_BINARY)
        _, thresh_contour = cv2.threshold(gray_contour, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours_gcode, _ = cv2.findContours(thresh_gcode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_contour, _ = cv2.findContours(thresh_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get main contours
        contour_gcode = max(contours_gcode, key=cv2.contourArea)
        contour_real = max(contours_contour, key=cv2.contourArea)
        
        # Calculate orientation and necessary rotation
        rect_gcode = cv2.minAreaRect(contour_gcode)
        rect_real = cv2.minAreaRect(contour_real)
        
        angle_gcode = rect_gcode[2]
        angle_real = rect_real[2]
        
        # Normalize angles
        if angle_gcode < -45:
            angle_gcode += 90
        if angle_real < -45:
            angle_real += 90
            
        rotation_angle = angle_gcode - angle_real
        
        print(f"G-code angle: {angle_gcode:.2f}°")
        print(f"Real contour angle: {angle_real:.2f}°")
        print(f"Required rotation: {rotation_angle:.2f}°")
        
        # Calculate centers
        x_g, y_g, w_g, h_g = cv2.boundingRect(contour_gcode)
        x_r, y_r, w_r, h_r = cv2.boundingRect(contour_real)
        
        cx_gcode = x_g + w_g/2
        cy_gcode = y_g + h_g/2
        cx_real = x_r + w_r/2
        cy_real = y_r + h_r/2
        
        # Create transformation matrix
        M = cv2.getRotationMatrix2D((cx_gcode, cy_gcode), rotation_angle, real_scale)
        
        # Calculate translation
        cx_gcode_transformed = M[0,0] * cx_gcode + M[0,1] * cy_gcode + M[0,2]
        cy_gcode_transformed = M[1,0] * cx_gcode + M[1,1] * cy_gcode + M[1,2]
        
        dx = cx_real - cx_gcode_transformed
        dy = cy_real - cy_gcode_transformed
        
        # Adjust transformation
        M[0,2] += dx
        M[1,2] += dy
        
        return M
        
    except Exception as e:
        print(f"Error in initial alignment: {str(e)}")
        return None

def process_gcode_overlay(segmented_image: np.ndarray, figure_contours: List[np.ndarray], 
                         output_dir: str, scale_factor: Optional[float] = None, 
                         gcode_scale_factor: Optional[float] = None, showSteps: bool = False, 
                         gcode_folder_path: Optional[str] = None, generate_gif: bool = True) -> Dict[str, Any]:
    """
    Processes all available G-code images and overlays them on the segmented image.
    
    Args:
        segmented_image: Segmented image with 3D figure contour
        figure_contours: List of detected contours of the 3D figure
        output_dir: Base directory to save results
        scale_factor: Scale factor in pixels per unit (optional)
        gcode_scale_factor: Pre-calculated G-code scale factor (optional)
        showSteps: Whether to show intermediate steps
        gcode_folder_path: Path to folder containing G-code images (optional)
        generate_gif: Whether to generate accumulation GIF (default True)
        
    Returns:
        Dictionary with processing information including paths of generated images,
        global transformation, and line data for subsequent analysis
    """
    print("\n--- Processing G-code line overlay ---")
    
    # Check if we have valid contours
    if figure_contours is None or len(figure_contours) == 0:
        print("Cannot perform overlay: No contours detected")
        return {"success": False, "error": "No contours detected"}
    
    # Select the largest contour (main) for processing
    centered_contour = max(figure_contours, key=cv2.contourArea)
    print(f"Main contour selected for overlay (area: {cv2.contourArea(centered_contour):.1f} px²)")
    
    # Separate internal contours
    internal_contours = [contour for contour in figure_contours if not np.array_equal(contour, centered_contour)]
    if internal_contours:
        print(f"Found {len(internal_contours)} additional internal contours")
    
    # Create directory for overlay
    overlay_dir = os.path.join(output_dir, "gcode_overlay")
    os.makedirs(overlay_dir, exist_ok=True)
    
    # Create directory for processed lines
    contours_dir = os.path.join(overlay_dir, "line_contours")
    os.makedirs(contours_dir, exist_ok=True)
    
    # Create reference image for alignment
    contour_image = np.zeros_like(segmented_image)
    cv2.drawContours(contour_image, [centered_contour], -1, (0, 0, 255), -1)  # Red fill
    cv2.drawContours(contour_image, [centered_contour], -1, (255, 255, 255), 2)  # White border
    
    # Use the provided path as a parameter
    gcode_dir = gcode_folder_path
    
    # Check if directory exists
    if not os.path.exists(gcode_dir):
        print(f"G-code directory not found: {gcode_dir}")
        return {"success": False, "error": f"Directory not found: {gcode_dir}"}
    
    # Find G-code images
    gcode_images = sorted(glob.glob(os.path.join(gcode_dir, "line_*.png")))
    
    if not gcode_images:
        print(f"No G-code images found in {gcode_dir}")
        return {"success": False, "error": "No G-code images found"}
    
    print(f"Found {len(gcode_images)} G-code images")
    
    # Get G-code scale factor only if not provided
    if gcode_scale_factor is None:
        try:
            first_gcode = gcode_images[0]
            gcode_scale_factor, _ = extract_scale_from_image(
                first_gcode, 
                output_dir=None,
                showSteps=False
            )
            print(f"Detected G-code scale: {gcode_scale_factor:.4f} pixels/mm")
        except Exception as e:
            print(f"Warning: Could not extract scale from G-code: {e}")
            gcode_scale_factor = None
    
    results = {
        "success": True,
        "processed_images": len(gcode_images),
        "lines": [],
        "scale_info": {
            "image_scale": scale_factor,
            "gcode_scale": gcode_scale_factor
        },
        "figure_contour": centered_contour,  # Save main contour
        "internal_contours": internal_contours,  # Save internal contours
        "global_transform": None,  # Will be filled with transformation matrix
    }
    
    # Step 1: Calculate global transformation
    print("\n--- Calculating global alignment ---")
    
    # Extract green line from first image
    first_gcode = gcode_images[0]
    first_mask, first_lines, first_image = extract_green_line(first_gcode, showSteps=False)
    
    # Calculate transformation
    print("Calculating global transformation...")
    
    global_transform = calculate_global_transformation(
        first_image, 
        contour_image, 
        scale_factor, 
        gcode_scale_factor,
        showSteps
    )
    
    if global_transform is None:
        return {"success": False, "error": "Failed to calculate transformation"}
    
    results["global_transform"] = global_transform.tolist()  # Convert to list for JSON
    print("Global transformation calculated successfully")
    
    # Step 2: Process each line individually
    print("\n--- Processing individual lines ---")
    
    # Target image dimensions
    h, w = segmented_image.shape[:2]
    
    # For accumulation and GIF
    gif_frames = []
    cumulative_result = segmented_image.copy()
    cumulative_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw contour on accumulated image
    cv2.drawContours(cumulative_result, [centered_contour], -1, (0, 0, 255), 2)
    
    for i, gcode_path in enumerate(gcode_images):
        gcode_name = os.path.basename(gcode_path)
        print(f"Processing line {i+1}/{len(gcode_images)}: {gcode_name}")
        
        try:
            # Extract green lines
            green_mask, green_lines, original_image = extract_green_line(
                gcode_path, 
                output_dir=None,
                showSteps=showSteps
            )
            
            # If no lines found, continue with next image
            if not green_lines:
                print(f"  No green lines found in {gcode_name}")
                continue
            
            # Draw transformed lines
            overlay_result, line_mask = draw_transformed_lines(
                lines=green_lines,
                transform_matrix=global_transform,
                target_image=segmented_image,
                contour=centered_contour,
                showSteps=showSteps
            )
            
            # Save individual result
            contour_file_name = f"line_contour_{i:03d}_{gcode_name}"
            contours_path = os.path.join(contours_dir, contour_file_name)
            cv2.imwrite(contours_path, overlay_result)
            
            # Update accumulated mask
            cumulative_mask = cv2.bitwise_or(cumulative_mask, line_mask)
            
            # Update accumulated image for GIF
            # Create fresh copy of segmented image
            cumulative_result = segmented_image.copy()
            # Draw figure contour
            cv2.drawContours(cumulative_result, [centered_contour], -1, (0, 0, 255), 2)
            # Draw all lines accumulated so far
            cumulative_result[cumulative_mask > 0] = (0, 255, 0)
            
            # Add frame to GIF
            rgb_overlay = cv2.cvtColor(cumulative_result.copy(), cv2.COLOR_BGR2RGB)
            gif_frames.append(rgb_overlay)
            
            # Transform the original points for later analysis
            transformed_line_data = process_line_transformation(green_lines, global_transform)
            
            # Register information
            line_result = {
                "line_index": i,
                "line_name": gcode_name,
                "contours_path": contours_path,
                "success": True,
                "line_mask": line_mask,  # Save mask directly, without converting to list
                "transformed_lines": transformed_line_data
            }
            results["lines"].append(line_result)
            
        except Exception as e:
            print(f"Error processing line {gcode_name}: {str(e)}")
            line_result = {
                "line_index": i,
                "line_name": gcode_name,
                "success": False,
                "error": str(e)
            }
            results["lines"].append(line_result)
    
    print(f"G-code processing completed. Results saved in: {overlay_dir}")
    
    # Include the cumulative result image in results
    results["cumulative_result"] = cumulative_result
    
    # Include frames for GIF to be created in main file if needed
    results["gif_frames"] = gif_frames
    
    # Save cumulative mask as image to recover it later if needed
    cumulative_mask_path = os.path.join(overlay_dir, "cumulative_mask.png") 
    cv2.imwrite(cumulative_mask_path, cumulative_mask)
    results["cumulative_mask_path"] = cumulative_mask_path
    
    return results

def process_line_transformation(green_lines: List[Dict[str, Any]], 
                              global_transform: np.ndarray) -> List[Dict[str, Any]]:
    """
    Process and transform line data for further analysis.
    
    Args:
        green_lines: List of detected green lines
        global_transform: Global transformation matrix
        
    Returns:
        List of dictionaries with transformed line data
    """
    transformed_line_data = []
    for line in green_lines:
        contour_points = line['contour'].reshape(-1, 2).astype(np.float32)
        transformed_points = cv2.transform(contour_points.reshape(-1, 1, 2), global_transform)
        transformed_contour = transformed_points.reshape(-1, 1, 2).astype(np.int32)
        
        approx_points = line['approx'].reshape(-1, 2).astype(np.float32)
        transformed_approx = cv2.transform(approx_points.reshape(-1, 1, 2), global_transform)
        transformed_approx = transformed_approx.reshape(-1, 1, 2).astype(np.int32)
        
        # Calculate transformed thickness
        thickness = max(1, int(line['thickness'] * global_transform[0, 0]))
        
        # Save data for analysis
        transformed_line_data.append({
            'original_contour': line['contour'].tolist(),
            'transformed_contour': transformed_contour.tolist(),
            'original_approx': line['approx'].tolist(),
            'transformed_approx': transformed_approx.tolist(),
            'original_thickness': line['thickness'],
            'transformed_thickness': thickness
        })
    
    return transformed_line_data

if __name__ == "__main__":
    gcode_image_path = "your_gcode_image_path_here.png"  # Replace with your image path if you want to test
    output_dir = "your_output_directory_here"  # Replace with your output directory if you want to test
    os.makedirs(output_dir, exist_ok=True)

    # Extract example green line
    green_mask, contours, original_image = extract_green_line(
        gcode_image_path,
        output_dir=output_dir,
        showSteps=True
    )
    
    print("Green line extraction completed.")