import cv2
import os
import sys
import datetime
import numpy as np
from typing import Dict, Tuple, List, Optional, Any

# Add parent directory to path to import functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required functions
from functions.scale_detection.scale_ruler import detect_ruler_scale
from functions.fix_distorsion.fix_distorsion import detect_distorsion
from functions.figure_detection.figure_detection import segment_3d_figure
from functions.reporting.pdf_generator import generate_report
from functions.overlay_gcode.overlay_lines import process_gcode_overlay
from functions.width_detection.analyze_lines import process_gcode_analysis

def setup_output_directory(image_path: str, base_output_dir: str = None) -> Tuple[str, str, str]:
    """
    Set up the output directory structure for saving results.
    
    Args:
        image_path: Path to the input image file
        base_output_dir: Optional base directory for output. If None, uses default location.
        
    Returns:
        Tuple containing image name, timestamp, and output directory path
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if base_output_dir is None:
        # Use parent directory/output
        base_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    
    output_dir = os.path.join(base_output_dir, image_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    return image_name, timestamp, output_dir

def detect_image_scale(image_path: str, output_dir: str, image_name: str, 
                      image_paths: Dict[str, str], showSteps=False) -> Optional[Dict[str, Any]]:
    """
    Detect scale from ruler in the image.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save output images
        image_name: Name of the image file without extension
        image_paths: Dictionary to store paths of generated images
        showSteps: Whether to display intermediate steps
        
    Returns:
        Dictionary containing scale information or None if detection fails
    """
    print("\n--- Detecting scale from ruler ---")
    original_scale, _, scale_image = detect_ruler_scale(image_path, showSteps=showSteps)
    
    if original_scale is None:
        print("Error: Could not detect scale in the image")
        return None
    
    # Save scale image
    scale_output_path = os.path.join(output_dir, f"01_scale_{image_name}.png")
    cv2.imwrite(scale_output_path, scale_image)
    image_paths[f"01_scale_{image_name}.png"] = scale_output_path
    
    print(f"Scale image saved to: {scale_output_path}")
    print(f"Detected scale: {original_scale['pixels_per_unit']:.2f} px/{original_scale['unit']}")
    
    return original_scale

def correct_image_distortion(image_path: str, original_scale: Dict[str, Any], 
                            output_dir: str, image_name: str,
                            image_paths: Dict[str, str], showSteps=False) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """
    Detect and correct image distortion.
    
    Args:
        image_path: Path to the input image
        original_scale: Dictionary containing original scale information
        output_dir: Directory to save output images
        image_name: Name of the image file without extension
        image_paths: Dictionary to store paths of generated images
        showSteps: Whether to display intermediate steps

    Returns:
        Tuple containing distortion info, adjusted scale, and path to corrected image
    """
    print("\n--- Detecting and correcting distortion ---")
    scale_factor = original_scale['pixels_per_unit']
    
    distortion, _, corrected_image, adjusted_scale, comparison_images = detect_distorsion(
        image_path, showSteps=showSteps, scale_factor=scale_factor, escala_original=original_scale, output_dir=output_dir
    )
    
    # Add comparison images to the image paths dictionary
    if comparison_images:
        for img_key, img_path in comparison_images.items():
            if img_key == "before_square":
                image_paths[f"02a_before_square_{image_name}.png"] = img_path
            elif img_key == "after_square":
                image_paths[f"02b_after_square_{image_name}.png"] = img_path
    
    if distortion is None:
        print("Error: Could not detect distortion in the image")
        return None, None, ""
    
    # Save corrected image
    corrected_output_path = os.path.join(output_dir, f"02_corrected_{image_name}.png")
    cv2.imwrite(corrected_output_path, corrected_image)
    image_paths[f"02_corrected_{image_name}.png"] = corrected_output_path
    
    print(f"Corrected image saved to: {corrected_output_path}")
    
    if adjusted_scale:
        print(f"Adjusted scale: {adjusted_scale['pixels_per_unit']:.2f} px/{adjusted_scale['unit']}")
    
    return distortion, adjusted_scale, corrected_output_path

def segment_figure(corrected_image_path: str, output_dir: str, image_name: str, 
                  image_paths: Dict[str, str],showSteps=False) -> Tuple[np.ndarray, List[Any]]:
    """
    Segment the 3D figure from the corrected image.
    
    Args:
        corrected_image_path: Path to the corrected image
        output_dir: Directory to save output images
        image_name: Name of the image file without extension
        image_paths: Dictionary to store paths of generated images
        showSteps: Whether to display intermediate steps
        
    Returns:
        Tuple containing segmented image and figure contours
    """
    print("\n--- Segmenting 3D Figure ---")
    
    # Load corrected image
    processed_image = cv2.imread(corrected_image_path)
    if processed_image is None:
        raise FileNotFoundError(f"Could not load corrected image from {corrected_image_path}")
    
    # Segment 3D figure
    segmented_image, figure_contours = segment_3d_figure(processed_image, showSteps=showSteps)
    
    # Save segmented image
    segmented_output_path = os.path.join(output_dir, f"03_segmented_{image_name}.png")
    cv2.imwrite(segmented_output_path, segmented_image)
    image_paths[f"03_segmented_{image_name}.png"] = segmented_output_path
    
    print(f"Segmented image saved to: {segmented_output_path}")
    return segmented_image, figure_contours

def process_gcode_visualization(segmented_image: np.ndarray, figure_contours: List[Any], 
                               output_dir: str, image_name: str, adjusted_scale: Dict[str, Any],
                               gcode_folder_path: str, generate_gif: bool, 
                               image_paths: Dict[str, str],showSteps=False) -> Dict[str, Any]:
    """
    Process G-code overlay and visualization.
    
    Args:
        segmented_image: Segmented image array
        figure_contours: Detected figure contours
        output_dir: Directory to save output images
        image_name: Name of the image file without extension
        adjusted_scale: Dictionary containing adjusted scale information
        gcode_folder_path: Path to folder containing G-code images
        generate_gif: Whether to generate an accumulation GIF
        image_paths: Dictionary to store paths of generated images
        showSteps: Whether to display intermediate steps

    Returns:
        Dictionary containing overlay results
    """
    print("\n--- Processing G-code overlay ---")
    
    overlay_results = process_gcode_overlay(
        segmented_image=segmented_image,
        figure_contours=figure_contours,
        output_dir=output_dir,
        scale_factor=adjusted_scale['pixels_per_unit'] if adjusted_scale else None,
        showSteps=showSteps,
        gcode_folder_path=gcode_folder_path,
        generate_gif=generate_gif
    )
    
    # Save the G-code overlay
    if overlay_results.get("success", False) and "cumulative_result" in overlay_results:
        overlay_image = overlay_results["cumulative_result"]
        overlay_output_path = os.path.join(output_dir, f"04_overlay_gcode_{image_name}.png")
        cv2.imwrite(overlay_output_path, overlay_image)
        image_paths[f"04_overlay_gcode_{image_name}.png"] = overlay_output_path
        print(f"G-code overlay saved to: {overlay_output_path}")
    
    # Generate and save the GIF if requested
    if generate_gif and overlay_results.get("success", False) and "gif_frames" in overlay_results:
        try:
            import imageio
            
            gif_frames = overlay_results["gif_frames"]
            
            if gif_frames:
                gif_output_path = os.path.join(output_dir, f"05_gcode_accumulation_{image_name}.gif")
                
                # Create the GIF
                imageio.mimsave(gif_output_path, gif_frames, duration=0.3)
                image_paths[f"05_gcode_accumulation_{image_name}.gif"] = gif_output_path
                print(f"G-code accumulation GIF saved to: {gif_output_path}")
                
                # Add the path to the results
                overlay_results["gif_path"] = gif_output_path
        except Exception as e:
            print(f"Error creating GIF: {str(e)}")

    return overlay_results    
  
def analyze_gcode_lines(overlay_results: Dict[str, Any], segmented_image: np.ndarray, 
                       output_dir: str, save_analysis_images: bool,
                       export_samples_csv: bool = False, 
                       num_samples: int = 20) -> Dict[str, Any]:
    """
    Analyze G-code lines and generate statistics.
    
    Args:
        overlay_results: Results from G-code overlay processing
        segmented_image: Segmented image array
        output_dir: Directory to save output images
        save_analysis_images: Whether to save analysis images
        export_samples_csv: Whether to export per-sample width measurements
        num_samples: Number of width samples to take per line in the sample csv
        
    Returns:
        Dictionary containing analysis results
    """
    print("\n--- Analyzing G-code lines ---")
    
    analysis_results = process_gcode_analysis(
        overlay_results=overlay_results,
        segmented_image=segmented_image,
        output_dir=output_dir,
        save_images=save_analysis_images,
        export_samples_csv=export_samples_csv,
        num_samples=num_samples
    )

    # Add output directory to results for reference in GUI
    analysis_results['output_dir'] = output_dir

    if analysis_results.get("success", False):
        print(f"Line analysis CSV: {analysis_results.get('csv_path', '')}")
        
        # Print key metrics from the analysis
        summary = analysis_results.get('summary', {})
        if summary:
            print("\nKey metrics:")
            print(f"  - Total lines analyzed: {summary.get('total_lines_analyzed', 0)}")
            print(f"  - Average width (mm): {summary.get('mean_width_mm', 0):.4f} mm")
            print(f"  - Average width variation: {summary.get('mean_width_variation', 0):.2f}%")
    else:
        print(f"Line analysis failed: {analysis_results.get('error', 'Unknown error')}")
    
    return analysis_results

def create_report(output_dir: str, image_name: str, gcode_folder_name: str, 
                 adjusted_scale: Dict[str, Any], original_scale: Dict[str, Any], 
                 overlay_results: Dict[str, Any], analysis_results: Dict[str, Any], 
                 image_paths: Dict[str, str], distortion_info: Dict[str, Any]) -> None:
    """
    Generate a PDF report of the analysis.
    
    Args:
        output_dir: Directory to save the report
        image_name: Name of the image file without extension
        gcode_folder_name: Name of the G-code folder
        adjusted_scale: Dictionary containing adjusted scale information
        original_scale: Dictionary containing original scale information
        overlay_results: Results from G-code overlay processing
        analysis_results: Results from G-code analysis
        image_paths: Dictionary containing paths of generated images
        distortion_info: Information about detected distortion
    """
    print("\n--- Generating PDF Report ---")
    
    try:
        report_path = generate_report(
            output_dir=output_dir, 
            image_name=image_name,
            gcode_folder_name=gcode_folder_name,
            scale_info=adjusted_scale if adjusted_scale else original_scale,
            overlay_results=overlay_results,
            analysis_results=analysis_results,
            image_paths=image_paths,
            original_scale_info=original_scale,
            distorsion_info=distortion_info
        )
        print(f"PDF report generated: {report_path}")
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        import traceback
        traceback.print_exc()

def main(image_path: str = None, gcode_folder_path: str = None, 
         save_analysis_images: bool = True, generate_gif: bool = True, 
         generate_report_pdf: bool = True, base_output_dir: str = None,
         export_samples_csv: bool = False, num_samples: int = 20,
         showSteps=False) -> Optional[Tuple[np.ndarray, Dict[str, Any], List[Any], Dict[str, Any]]]:
    """
    Main application flow for processing 3D template images.
    
    Args:
        image_path: Path to the image file to be processed
        gcode_folder_path: Path to the folder containing G-code images
        save_analysis_images: Whether to save analysis images
        generate_gif: Whether to generate the accumulation GIF
        generate_report_pdf: Whether to generate a PDF report
        base_output_dir: Optional base directory for output. If None, uses default location.
        export_samples_csv: Whether to export per-sample width measurements
        num_samples: Number of width samples to take per line in the sample csv
        showSteps: Whether to display intermediate steps

        
    Returns:
        Tuple containing processed image, adjusted scale, figure contours, and analysis results
    """
    # Verify image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        return None
    
    # Setup output directory
    image_name, timestamp, output_dir = setup_output_directory(image_path, base_output_dir)
    
    # Get G-code folder name
    gcode_folder_name = os.path.basename(gcode_folder_path) if gcode_folder_path else "unknown"
    
    print(f"\nProcessing image: {image_path}")
    print(f"Saving results to: {output_dir}")
    
    # Dictionary to store all generated image paths
    image_paths = {}
    
    # Step 1: Detect scale from ruler
    original_scale = detect_image_scale(image_path, output_dir, image_name, image_paths, showSteps=showSteps)
    if original_scale is None:
        return None
    
    # Step 2: Detect and fix distortion
    distortion, adjusted_scale, corrected_output_path = correct_image_distortion(
        image_path, original_scale, output_dir, image_name, image_paths, showSteps=showSteps
    )
    if distortion is None:
        return None
    
    # Step 3: Segment 3D figure
    try:
        segmented_image, figure_contours = segment_figure(corrected_output_path, output_dir, image_name, image_paths,showSteps=showSteps)
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        return None
    
    # Step 4: Process G-code overlay
    overlay_results = process_gcode_visualization(
        segmented_image, figure_contours, output_dir, image_name, 
        adjusted_scale, gcode_folder_path, generate_gif, image_paths,
        showSteps=showSteps
    )
    
    # Step 5: Analyze lines and measure statistics
    analysis_results = analyze_gcode_lines(
        overlay_results, segmented_image, output_dir, save_analysis_images,
        export_samples_csv=export_samples_csv, num_samples=num_samples
    )

    print("\n--- Image processing completed ---")
    print(f"All results saved to: {output_dir}")
    
    # Step 6: Generate PDF report if requested
    if generate_report_pdf:
        create_report(
            output_dir, image_name, gcode_folder_name, adjusted_scale, original_scale,
            overlay_results, analysis_results, image_paths, distortion
        )
    
    # Return processed image, adjusted scale, contours, and analysis results for further processing
    processed_image = cv2.imread(corrected_output_path)
    return processed_image, adjusted_scale, figure_contours, analysis_results

if __name__ == "__main__":

    image_path = "your_image_path_here.png"  # Replace with your image path if you want to run this script directly
    gcode_folder_path = "your_gcode_folder_path_here"  # Replace with your G-code folder path if you want to run this script directly

    main(image_path=image_path, gcode_folder_path=gcode_folder_path, 
         save_analysis_images=True, generate_gif=True, generate_report_pdf=True, showSteps=True)