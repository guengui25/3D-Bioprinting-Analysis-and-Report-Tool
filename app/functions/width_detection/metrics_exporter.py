import os
import csv
import numpy as np
from typing import List, Dict, Optional, Any

def generate_summary(analyzed_lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates a statistical summary of all analyzed lines.
    
    Args:
        analyzed_lines: List of line analysis results
        
    Returns:
        Statistical summary dictionary
    """
    # Filter only lines with successful analysis
    successful_lines = [line for line in analyzed_lines if line.get("success", False)]
    
    if not successful_lines:
        return {"total_lines_analyzed": 0}
    
    # Extract width statistics
    widths = [line.get("width_analysis", {}).get("width_pixels", 0) for line in successful_lines]
    widths_mm = [line.get("width_analysis", {}).get("width_mm", 0) for line in successful_lines]
    expected_widths = [line.get("width_analysis", {}).get("expected_width_pixels", 0) for line in successful_lines]
    std_widths = [line.get("width_analysis", {}).get("std_width_pixels", 0) for line in successful_lines]
    width_variations = [line.get("width_analysis", {}).get("width_variation_percentage", 0) for line in successful_lines]
    
    # Calculate global statistics
    summary = {
        "total_lines_analyzed": len(successful_lines),
        "mean_width_pixels": np.mean(widths) if widths else 0,
        "mean_width_mm": np.mean(widths_mm) if widths_mm else 0,
        "mean_expected_width_pixels": np.mean(expected_widths) if expected_widths else 0,
        "mean_std_width_pixels": np.mean(std_widths) if std_widths else 0,
        "mean_width_variation": np.mean(width_variations) if width_variations else 0,
    }
    
    return summary

def save_results_to_csv(line_results: List[Dict[str, Any]], output_dir: str, 
                       scale_factor: Optional[float] = None) -> str:
    """
    Saves line analysis results to a CSV file.
    
    Args:
        line_results: List of line analysis results
        output_dir: Directory to save results
        scale_factor: Scale factor in pixels per mm
        
    Returns:
        Path to the CSV file
    """
    # Create CSV for results
    csv_path = os.path.join(output_dir, "line_results.csv")
    
    with open(csv_path, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write header with updated column names
        csv_writer.writerow([
            "Line name", "Width (px)", "Width (mm)",
            "Standard deviation (px)", "Standard deviation (mm)",
            "Maximum width (mm)", "Minimum width (mm)", "Width variation (%)",
            "Crosses contours"
        ])
        
        # Add data for each line
        for result in line_results:
            if not result.get("success", False):
                continue
                
            line_name = result.get("line_name", "unnamed_line")
            width_data = result.get("width_analysis", {})
            
            # Calculate max and min width in mm if not exist
            radius_values = width_data.get("radius_values", [])
            if scale_factor and scale_factor > 0:
                width_max_mm = width_data.get("width_max_mm", 0)
                width_min_mm = width_data.get("width_min_mm", 0)
                if width_max_mm == 0 and radius_values:
                    width_max_mm = (2 * np.max(radius_values)) / scale_factor
                if width_min_mm == 0 and radius_values:
                    width_min_mm = (2 * np.min(radius_values)) / scale_factor
            else:
                width_max_mm = width_data.get("width_max_mm", 0)
                width_min_mm = width_data.get("width_min_mm", 0)
            
            # Use std_width_mm directly instead of converting to cm
            std_width_mm = width_data.get("std_width_mm", 0)
            
            # Get the flag if line crosses contours
            crosses_contours = "Yes" if result.get("crosses_contours", False) else "No"
            
            csv_writer.writerow([
                line_name,
                f"{width_data.get('width_pixels', 0):.2f}", 
                f"{width_data.get('width_mm', 0):.4f}",
                f"{width_data.get('std_width_pixels', 0):.2f}",
                f"{std_width_mm:.4f}",
                f"{width_max_mm:.4f}",
                f"{width_min_mm:.4f}",
                f"{width_data.get('width_variation_percentage', 0):.2f}",
                crosses_contours
            ])
    
    return csv_path

def save_width_samples_to_csv(line_results: List[Dict[str, Any]], output_dir: str, scale_factor: Optional[float] = None, num_samples: int = 20) -> str:
    """
    Saves per-sample width measurements for each line to a CSV file.
    Limited to the specified number of samples for readability.
    
    Args:
        line_results: List of line analysis results
        output_dir: Directory to save results
        scale_factor: Scale factor in pixels per mm
        num_samples: Number of samples to include in the CSV
    """
    csv_path = os.path.join(output_dir, "lines_width_details.csv")
    
    unit = "mm" if scale_factor and scale_factor > 0 else "px"
    with open(csv_path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header: one column per sample in proper unit
        header = ["Line name"] + [f"Sample_{i+1} ({unit})" for i in range(num_samples)]
        writer.writerow(header)
        
        for result in line_results:
            line_name = result.get("line_name", "unnamed_line")
            samples_px = result.get("width_analysis", {}).get("width_samples", [])
            
            # Subsample the measurements to the requested number
            if len(samples_px) > num_samples:
                # Use linear indexing to get evenly spaced samples
                if len(samples_px) > 1:
                    indices = np.linspace(0, len(samples_px) - 1, num_samples, dtype=int)
                    samples_px = [samples_px[i] for i in indices]
                else:
                    # Just duplicate the single sample
                    samples_px = samples_px * num_samples
            
            # Convert pixel measurements to mm if scale_factor provided
            if scale_factor and scale_factor > 0:
                samples_mm = [w / scale_factor for w in samples_px]
            else:
                samples_mm = [w for w in samples_px]
            
            # Format measurements and pad missing values
            row = [line_name] + [f"{mm:.4f}" for mm in samples_mm] + [""] * (num_samples - len(samples_mm))
            writer.writerow(row)
    
    return csv_path

def export_all_metrics(line_results: List[Dict[str, Any]], output_dir: str, 
                      num_samples: int = 20, scale_factor: Optional[float] = None,
                      export_samples_csv: bool = False) -> Dict[str, Any]:
    """
    Exports all metrics and generates analysis charts.
    
    Args:
        line_results: List of line analysis results
        output_dir: Directory to save results
        scale_factor: Scale factor in pixels per mm
        
    Returns:
        Dictionary with information about exported files
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate summary (for internal use only)
    summary = generate_summary(line_results)
    
    # Save results to CSV
    csv_path = save_results_to_csv(line_results, output_dir, scale_factor)

    # Save per-sample measurements if requested
    if export_samples_csv:
        samples_csv_path = save_width_samples_to_csv(line_results, output_dir, scale_factor, num_samples)
        print(f"Width details CSV file saved at: {samples_csv_path}")
    else:
        samples_csv_path = None

    print(f"\nMetrics exported. CSV file saved at: {csv_path}")

    return {
        "success": True,
        "summary": summary,
        "csv_path": csv_path,
        "samples_csv_path": samples_csv_path
    }