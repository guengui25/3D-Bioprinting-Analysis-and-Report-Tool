import os
import datetime
import pandas as pd
from fpdf import FPDF
import glob
import cv2
import numpy as np
from typing import Dict, Optional, Any, List, Tuple, Union

class CustomPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, '3D Print Analysis Report', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

def generate_report(output_dir: str, image_name: str, gcode_folder_name: str, 
                   scale_info: Dict[str, Any], overlay_results: Dict[str, Any], 
                   analysis_results: Dict[str, Any], image_paths: Dict[str, str], 
                   original_scale_info: Optional[Dict[str, Any]] = None, 
                   distorsion_info: Optional[Dict[str, Any]] = None) -> str:
    """Generates a complete PDF report with analysis results.
    
    Args:
        output_dir: Directory to save the PDF
        image_name: Name of the analyzed image
        gcode_folder_name: Name of the G-code folder
        scale_info: Corrected scale information
        overlay_results: Overlay processing results
        analysis_results: Line analysis results
        image_paths: Paths to generated images
        original_scale_info: Original scale information (before correction)
        distorsion_info: Information about detected distortion
        
    Returns:
        Path to the generated PDF file
    """
    # Initialize PDF
    pdf = CustomPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Generate the different sections of the report
    add_general_information_section(pdf, image_name, gcode_folder_name, 
                                   analysis_results, overlay_results, scale_info)
    add_results_table_section(pdf, analysis_results)
    add_processing_images_section(pdf, image_paths, original_scale_info, scale_info, 
                                 distorsion_info, output_dir)
    add_line_analysis_appendix(pdf, analysis_results, output_dir, image_paths)
    
    # Save PDF
    report_path = os.path.join(output_dir, f"analysis_report_{image_name}.pdf")
    pdf.output(report_path)
    
    return report_path

def add_general_information_section(pdf: FPDF, image_name: str, gcode_folder_name: str,
                                  analysis_results: Dict[str, Any], overlay_results: Dict[str, Any],
                                  scale_info: Dict[str, Any]) -> None:
    """Adds general information section to the PDF report.
    
    Args:
        pdf: FPDF object to add content to
        image_name: Name of the analyzed image
        gcode_folder_name: Name of the G-code folder
        analysis_results: Results from line analysis
        overlay_results: Results from overlay processing
        scale_info: Information about image scale
        
    Returns:
        None
    """
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'General Information', 0, 1)
    pdf.ln(2)
    
    # Date and time
    pdf.set_font('Arial', '', 11)
    current_datetime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    pdf.cell(0, 8, f'Date and time: {current_datetime}', 0, 1)
    
    # Analyzed image
    pdf.cell(0, 8, f'Analyzed image: {image_name}', 0, 1)
    
    # G-code folder
    pdf.cell(0, 8, f'G-code used: {gcode_folder_name}', 0, 1)
    
    # Number of lines analyzed
    lines_analyzed = analysis_results.get("analyzed_lines", 0)
    pdf.cell(0, 8, f'Lines analyzed: {lines_analyzed}', 0, 1)
    
    # Number of detected contours
    internal_contours = len(overlay_results.get("internal_contours", []))
    total_contours = internal_contours + 1  # +1 for external contour
    pdf.cell(0, 8, f'Contours detected: {total_contours} ({internal_contours} internal)', 0, 1)
    
    # Detected scale with value
    scale_unit = scale_info.get('unit', 'mm')
    scale_value = scale_info.get('pixels_per_unit', 0)
    pdf.cell(0, 8, f'Detected scale: {scale_value:.2f} pixels/{scale_unit}', 0, 1)
    pdf.ln(5)

def add_results_table_section(pdf: FPDF, analysis_results: Dict[str, Any]) -> None:
    """Adds results table section to the PDF report.
    
    Args:
        pdf: FPDF object to add content to
        analysis_results: Results from line analysis
        
    Returns:
        None
    """
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Line Analysis Results', 0, 1)
    
    # Load CSV results
    csv_path = analysis_results.get("csv_path", "")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            
            # Store crossing lines information before dropping the column
            crossing_lines = []
            if "Crosses contours" in df.columns:
                crossing_lines_df = df[df["Crosses contours"] == "Yes"]
                if not crossing_lines_df.empty:
                    crossing_lines = crossing_lines_df["Line name"].tolist()
                df = df.drop(columns=["Crosses contours"])
            
            # Remove pixel-based measurements
            if "Width (px)" in df.columns:
                df = df.drop(columns=["Width (px)"])
            if "Standard deviation (px)" in df.columns:
                df = df.drop(columns=["Standard deviation (px)"])
            
            # Configure table to occupy full page width
            pdf.set_font('Arial', 'B', 6)
            
            # Get effective page width (minus margins)
            page_width = pdf.w - 2*pdf.l_margin
            
            # Calculate column widths based on CSV content
            col_widths_percent = {
                "Line name": 0.20,
                "Width (mm)": 0.16,
                "Standard deviation (mm)": 0.16,
                "Maximum width (mm)": 0.16,
                "Minimum width (mm)": 0.16,
                "Width variation (%)": 0.16
            }
            
            # Map column names to abbreviations for better visualization
            header_abbr = {
                "Line name": "Line name",
                "Width (mm)": "Width (mm)",
                "Standard deviation (mm)": "Std (mm)",
                "Maximum width (mm)": "Max (mm)",
                "Minimum width (mm)": "Min (mm)",
                "Width variation (%)": "Var (%)"
            }
            
            # Get column names from CSV (filtered without pixel measurements)
            headers = list(df.columns)
            
            # Calculate specific widths for each column
            col_widths = []
            for header in headers:
                # Look for corresponding width or use default
                if header in col_widths_percent:
                    width = page_width * col_widths_percent[header]
                else:
                    width = page_width * 0.15  # Default value
                col_widths.append(width)
            
            # Set smaller cell margins
            pdf.set_line_width(0.1)
            
            # Headers - with increased height to 14
            for i, header in enumerate(headers):
                if i < len(col_widths):
                    # Use abbreviation or original name
                    header_display = header_abbr.get(header, header)
                    pdf.cell(col_widths[i], 14, header_display, 1, 0, 'C')
            pdf.ln()
            
            # Data
            pdf.set_font('Arial', '', 6.5)  # Font for data
            for _, row in df.iterrows():
                for i, col in enumerate(headers):
                    if i < len(col_widths):
                        value = str(row[col])
                        # Format numbers for better visualization
                        if i == 0:  # Name column
                            # Remove .png extension if exists
                            if value.endswith(".png"):
                                value = value[:-4]
                        elif "." in value:  # Decimal numbers
                            try:
                                num = float(value)
                                if num >= 100:
                                    value = f"{num:.1f}"
                                elif num >= 10:
                                    value = f"{num:.2f}"
                                else:
                                    value = f"{num:.3f}"
                            except ValueError:
                                pass  # Keep original value if not a number
                        
                        # Determine if row should be colored red (when crossing contours)
                        line_name = str(row["Line name"])
                        is_crossing_line = line_name in crossing_lines
                        
                        if is_crossing_line:
                            pdf.set_text_color(194, 0, 0)  # Red text
                        else:
                            pdf.set_text_color(0, 0, 0)  # Normal black text
                            
                        pdf.cell(col_widths[i], 10, value, 1, 0, 'C')
                
                # Reset text color for next row
                pdf.set_text_color(0, 0, 0)
                pdf.ln()
            
            # Add crossing lines section after the table
            if crossing_lines:
                pdf.ln(10)
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 8, 'Lines that were not correctly printed:', 0, 1)
                
                pdf.set_font('Arial', '', 9)
                pdf.set_text_color(194, 0, 0)  # Red color for this section
                
                # Arrange lines in rows for better visualization
                max_lines_per_row = 4
                line_text = ""
                for i, line_name in enumerate(crossing_lines):
                    # Use full name (only remove .png extension if exists)
                    formatted_name = line_name
                    if formatted_name.endswith(".png"):
                        formatted_name = formatted_name[:-4]
                    
                    line_text += formatted_name
                    
                    # Add comma or period as needed
                    if i < len(crossing_lines) - 1:
                        if (i + 1) % max_lines_per_row == 0:
                            line_text += "."
                            pdf.cell(0, 6, line_text, 0, 1)
                            line_text = ""
                        else:
                            line_text += ", "
                
                # Print the last line if there's content
                if line_text:
                    pdf.cell(0, 6, line_text + ".", 0, 1)
                
                # Reset text color
                pdf.set_text_color(0, 0, 0)
            
        except Exception as e:
            pdf.set_font('Arial', '', 11)
            pdf.cell(0, 10, f'Error loading CSV data: {str(e)}', 0, 1)
    else:
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 10, 'Results CSV file not found', 0, 1)

def add_processing_images_section(pdf: FPDF, image_paths: Dict[str, str], 
                                original_scale_info: Optional[Dict[str, Any]], 
                                scale_info: Dict[str, Any],
                                distortion_info: Optional[Dict[str, Any]],
                                output_dir: str) -> None:
    """Adds processing images section to the PDF report.
    
    Args:
        pdf: FPDF object to add content to
        image_paths: Dictionary of image paths
        original_scale_info: Original scale information before correction
        scale_info: Current scale information
        distortion_info: Information about detected distortion
        output_dir: Directory for output files
        
    Returns:
        None
    """
    pdf.add_page()  # New page for processing images
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Processing Images', 0, 1, 'L')
    
    # Define precise order based on main_mask.py
    image_order = [
        '01_scale_',                # Step 1: Scale detection
        '02_corrected_',            # Step 2: Distortion correction
        '03_segmented_',            # Step 3: Figure segmentation
        '04_overlay_gcode_',        # Step 4: G-code overlay
        '05_gcode_accumulation_'    # Step 5: Accumulation GIF (optional)
    ]
    
    current_page = 1  # Start with first page of images
    
    # Iterate through images in the defined order
    for prefix in image_order:
        matching_images = [key for key in 
        image_paths.keys() if key.startswith(prefix)]
        for img_key in matching_images:
            # Change page after distortion correction
            if prefix == '03_segmented_' and current_page == 1:
                pdf.add_page()
                current_page = 2
                pdf.set_font('Arial', 'B', 14)
            
            img_path = image_paths[img_key]
            if os.path.exists(img_path) and not img_path.endswith('.gif'):  # Exclude GIFs, only process images
                # Add appropriate section based on image type
                if prefix == '01_scale_':
                    add_scale_detection_section(pdf, img_path, img_key, original_scale_info, scale_info)
                elif prefix == '02_corrected_':
                    add_distortion_correction_section(
                        pdf, img_path, img_key, original_scale_info, scale_info, 
                        distortion_info, image_paths
                    )
                elif prefix == '03_segmented_':
                    add_segmentation_section(pdf, img_path, img_key, output_dir)
                elif prefix == '04_overlay_gcode_':
                    add_gcode_overlay_section(pdf, img_path, img_key, output_dir)
                else:
                    # For other images that don't need special processing
                    add_generic_image_section(pdf, img_path, img_key)

def add_scale_detection_section(pdf: FPDF, img_path: str, img_key: str, 
                              original_scale_info: Optional[Dict[str, Any]], 
                              scale_info: Dict[str, Any]) -> None:
    """Adds scale detection section to the PDF.
    
    Args:
        pdf: FPDF object to add content to
        img_path: Path to the scale detection image
        img_key: Image key identifier
        original_scale_info: Original scale information before correction
        scale_info: Current scale information
        
    Returns:
        None
    """
    pdf.set_font('Arial', 'B', 11)
    title = 'Scale Detection'
    
    pdf.cell(0, 8, title, 0, 1, 'L')
    
    # Calculate center point of the page
    page_width = pdf.w - 2*pdf.l_margin
    image_width = 170  # Reduced size for the image
    x_centered = pdf.l_margin + (page_width - image_width)/2  # Center image
    
    # Add centered image
    pdf.image(img_path, x=x_centered, w=image_width)
    
    # Add caption with exact filename
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(0, 6, f'File: {img_key}', 0, 1, 'C')
    pdf.ln(5)
    
    # Add scale information with actual values
    pdf.set_font('Arial', '', 10)
    if original_scale_info:
        original_scale_unit = original_scale_info.get('unit', 'mm')
        original_scale_value = original_scale_info.get('pixels_per_unit', 0)
        pdf.cell(0, 6, f'Original detected scale: {original_scale_value:.2f} pixels/{original_scale_unit}', 0, 1)
    else:
        # If no original scale, use the corrected one
        scale_unit = scale_info.get('unit', 'mm')
        scale_value = scale_info.get('pixels_per_unit', 0)
        pdf.cell(0, 6, f'Detected scale: {scale_value:.2f} pixels/{scale_unit}', 0, 1)
    pdf.ln(3)

def add_distortion_correction_section(pdf: FPDF, img_path: str, img_key: str, 
                                    original_scale_info: Optional[Dict[str, Any]], 
                                    scale_info: Dict[str, Any], 
                                    distortion_info: Optional[Dict[str, Any]],
                                    image_paths: Dict[str, str]) -> None:
    """Adds distortion correction section to the PDF.
    
    Args:
        pdf: FPDF object to add content to
        img_path: Path to the distortion correction image
        img_key: Image key identifier
        original_scale_info: Original scale information before correction
        scale_info: Current scale information
        distortion_info: Information about detected distortion
        image_paths: Dictionary of all image paths
        
    Returns:
        None
    """
    pdf.set_font('Arial', 'B', 11)
    title = 'Distortion Correction'
    
    pdf.cell(0, 8, title, 0, 1, 'L')
    
    # Add detailed information about distortion correction
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, 'Distortion correction has been applied to the original image.', 0, 1)
    
    # Add details about the difference between original and corrected scale
    if original_scale_info:
        original_scale_unit = original_scale_info.get('unit', 'mm')
        original_scale_value = original_scale_info.get('pixels_per_unit', 0)
        
        # Calculate and show percentage change in scale
        scale_px_mm = scale_info.get('pixels_per_unit', 0)
        original_scale_px_mm = original_scale_info.get('pixels_per_unit', 0)
        if original_scale_px_mm > 0:
            scale_change_pct = ((scale_px_mm - original_scale_px_mm) / original_scale_px_mm) * 100
            pdf.cell(0, 6, f'Scale change after correction: {scale_change_pct:.2f}%', 0, 1)
        
        # Add information about corrected distortion with scale values
        pdf.cell(0, 6, f'Original scale: {original_scale_value:.2f} pixels/{original_scale_unit}', 0, 1)
        pdf.cell(0, 6, f'Corrected scale: {scale_px_mm:.2f} pixels/{scale_info.get("unit", "mm")}', 0, 1)
    
    # Look for before and after square images
    before_key = next((k for k in image_paths.keys() if '02a_before_square' in k), None)
    after_key = next((k for k in image_paths.keys() if '02b_after_square' in k), None)
    
    if before_key and after_key:
        pdf.ln(2)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, 'Reference Square - Before and After Correction:', 0, 1)
        
        # Calculate dimensions for image table (more compact)
        page_width = pdf.w - 2*pdf.l_margin
        img_width = page_width * 0.33  # Reduced from 35% to 33%
        
        # Create table without visible borders
        pdf.ln(1)
        
        # Row 1: Column headers
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(page_width/2, 4, 'Before', 0, 0, 'C')
        pdf.cell(page_width/2, 4, 'After', 0, 1, 'C')
        
        # Row 2: Images - ensure same height
        y_start = pdf.get_y()  # Save initial Y position
        
        # Load both images to determine appropriate height for alignment
        before_img = cv2.imread(image_paths[before_key])
        after_img = cv2.imread(image_paths[after_key])
        
        if before_img is not None and after_img is not None:
            before_ratio = before_img.shape[0] / before_img.shape[1]
            after_ratio = after_img.shape[0] / after_img.shape[1]
            
            # Use the same height for both images (based on average aspect ratio)
            avg_ratio = (before_ratio + after_ratio) / 2
            img_height = img_width * avg_ratio
        else:
            img_height = img_width  # Default to square if cannot load images
        
        # Create cell with sufficient height for image
        pdf.cell(page_width/2, img_height, '', 0, 0, 'C')
        pdf.cell(page_width/2, img_height, '', 0, 1, 'C')
        
        # Insert centered images in cells at exactly the same Y coordinate
        x_before = pdf.l_margin + (page_width/4) - (img_width/2)
        x_after = pdf.l_margin + (page_width*3/4) - (img_width/2)
        
        pdf.image(image_paths[before_key], x=x_before, y=y_start, w=img_width, h=img_height)
        pdf.image(image_paths[after_key], x=x_after, y=y_start, w=img_width, h=img_height)
        
        # Shared caption (more compact)
        pdf.set_font('Arial', 'I', 7)
        pdf.cell(0, 3, f'Files: {os.path.basename(image_paths[before_key])} | {os.path.basename(image_paths[after_key])}', 0, 1, 'C')
    
    pdf.ln(2)

def add_segmentation_section(pdf: FPDF, img_path: str, img_key: str, output_dir: str) -> None:
    """Adds 3D figure segmentation section to the PDF.
    
    Args:
        pdf: FPDF object to add content to
        img_path: Path to the segmentation image
        img_key: Image key identifier
        output_dir: Directory for output files
        
    Returns:
        None
    """
    pdf.set_font('Arial', 'B', 11)
    title = '3D Figure Segmentation'
    
    pdf.cell(0, 8, title, 0, 1, 'L')
    
    # Load and crop image for segmentation section
    img = cv2.imread(img_path)
    if img is not None:
        height, width, _ = img.shape
        
        # Vertical crop: remove top and bottom third
        crop_height = height // 3
        
        # Horizontal crop: remove 15% from each side
        crop_width = int(width * 0.15)
        
        # Crop image both vertically and horizontally
        cropped_img = img[crop_height:height-crop_height, crop_width:width-crop_width, :]
        
        # Save cropped image temporarily
        temp_path = os.path.join(output_dir, f"temp_cropped_{img_key}")
        cv2.imwrite(temp_path, cropped_img)
        
        # Calculate page center point
        page_width = pdf.w - 2*pdf.l_margin
        image_width = 140  # Reduced from 160 to 140 for better fit
        x_centered = pdf.l_margin + (page_width - image_width) / 2
        
        pdf.image(temp_path, x=x_centered, w=image_width)
        
        # Add caption
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 6, f'File: {img_key} (cropped)', 0, 1, 'C')
        pdf.ln(8)
        
        # Delete temporary file
        os.remove(temp_path)
    else:
        # If image can't be loaded, show error message
        pdf.cell(0, 8, title, 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, 'Error loading segmentation image.', 0, 1)
        pdf.ln(3)

def add_gcode_overlay_section(pdf: FPDF, img_path: str, img_key: str, output_dir: str) -> None:
    """Adds G-code overlay section to the PDF.
    
    Args:
        pdf: FPDF object to add content to
        img_path: Path to the G-code overlay image
        img_key: Image key identifier
        output_dir: Directory for output files
        
    Returns:
        None
    """
    pdf.set_font('Arial', 'B', 11)
    title = 'G-code Overlay'
    
    pdf.cell(0, 8, title, 0, 1, 'L')
    
    # Load and crop image for overlay section
    img = cv2.imread(img_path)
    if img is not None:
        height, width, _ = img.shape
        
        # Vertical crop: remove top and bottom third
        crop_height = height // 3
        
        # Horizontal crop: remove 15% from each side
        crop_width = int(width * 0.15)
        
        # Crop image both vertically and horizontally
        cropped_img = img[crop_height:height-crop_height, crop_width:width-crop_width, :]
        
        # Save cropped image temporarily
        temp_path = os.path.join(output_dir, f"temp_cropped_{img_key}")
        cv2.imwrite(temp_path, cropped_img)
        
        # Calculate page center point
        page_width = pdf.w - 2*pdf.l_margin
        image_width = 140  # Reduced from 160 to 140 for better fit
        x_centered = pdf.l_margin + (page_width - image_width) / 2 
        
        pdf.image(temp_path, x=x_centered, w=image_width)
        
        # Add caption
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 6, f'File: {img_key} (cropped)', 0, 1, 'C')
        pdf.ln(8)
        
        # Delete temporary file
        os.remove(temp_path)
    else:
        # If image can't be loaded, show error message
        pdf.cell(0, 8, title, 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, 'Error loading G-code overlay image.', 0, 1)
        pdf.ln(3)

def add_generic_image_section(pdf: FPDF, img_path: str, img_key: str) -> None:
    """Adds a generic image section to the PDF.
    
    Args:
        pdf: FPDF object to add content to
        img_path: Path to the image
        img_key: Image key identifier
        
    Returns:
        None
    """
    # Create title from image key
    title = img_key.replace('_', ' ').replace('.png', '')
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, title, 0, 1, 'L')
    
    # Center the image
    page_width = pdf.w - 2*pdf.l_margin
    image_width = 190
    x_centered = pdf.l_margin + (page_width - image_width)/2
    
    pdf.image(img_path, x=x_centered, w=image_width)
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(0, 6, f'File: {img_key}', 0, 1, 'C')
    pdf.ln(5)

def add_line_analysis_appendix(pdf: FPDF, analysis_results: Dict[str, Any], output_dir: str, 
                              image_paths: Dict[str, str]) -> None:
    """Adds line analysis appendix to the PDF report.
    
    Args:
        pdf: FPDF object to add content to
        analysis_results: Results from line analysis
        output_dir: Directory for output files
        image_paths: Dictionary of image paths
        
    Returns:
        None
    """
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Appendix: Individual Line Analysis', 0, 1, 'L')
    pdf.ln(2)  # Reduced space after title
    
    # Find line analysis images
    analysis_dir = os.path.join(output_dir, "line_analysis", "images")
    if os.path.exists(analysis_dir):
        analysis_images = sorted(glob.glob(os.path.join(analysis_dir, "analysis_*.png")))
        
        # Load crossing lines information from CSV
        crossing_lines = load_crossing_lines_from_csv(analysis_results.get("csv_path", ""))
        
        if analysis_images:
            # Configuration for 2-column layout
            images_per_row = 2
            images_per_page = 8  # 4 rows of 2 images
            margin = 5  # Reduced margin from 10 to 5
            page_width = pdf.w - 2*pdf.l_margin
            col_width = page_width / images_per_row
            img_width = col_width - 2*margin  # Image width with margin
            
            # Process images in rows
            for i in range(0, len(analysis_images), images_per_row):
                # Create new page if needed
                if i > 0 and i % images_per_page == 0:
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 14)
                    pdf.cell(0, 10, 'Individual Line Analysis (continued)', 0, 1, 'L')
                    pdf.ln(2)  # Reduced spacing
                
                # Get images for this row
                row_images = analysis_images[i:i+images_per_row]
                row_temp_paths = []
                row_heights = []
                
                # Step 1: process all images in row and get heights
                for img_path in row_images:
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Crop image
                        height, width, _ = img.shape
                        crop_height = height // 3
                        crop_width = int(width * 0.15)
                        cropped_img = img[crop_height:height-crop_height, crop_width:width-crop_width, :]
                        
                        # Save temporarily
                        temp_path = os.path.join(output_dir, f"temp_row_{os.path.basename(img_path)}")
                        cv2.imwrite(temp_path, cropped_img)
                        row_temp_paths.append(temp_path)
                        
                        # Calculate height preserving aspect ratio
                        cropped_height, cropped_width = cropped_img.shape[:2]
                        aspect_ratio = cropped_height / cropped_width
                        row_heights.append(img_width * aspect_ratio)
                    else:
                        row_temp_paths.append(None)
                        row_heights.append(img_width)  # Default height if image fails
                
                # Get maximum height for this row
                if row_heights:
                    max_height = max(row_heights)
                else:
                    max_height = img_width
                
                # Get current Y position for this row
                row_y = pdf.get_y()
                
                # Step 2: place all images with same height
                for j, temp_path in enumerate(row_temp_paths):
                    if temp_path is not None:
                        # X position to center in column
                        x = pdf.l_margin + j * col_width + (col_width - img_width) / 2
                        
                        # Place image with row's maximum height
                        pdf.image(temp_path, x=x, y=row_y, w=img_width, h=max_height)
                        
                        # Delete temporary file
                        os.remove(temp_path)
                
                # Advance cursor after images
                pdf.set_y(row_y + max_height + 2)
                
                # Step 3: add captions for all images
                for j, img_path in enumerate(row_images):
                    if img_path:
                        # Extract line name and remove .png but keep underscores
                        img_name = os.path.basename(img_path).replace('analysis_', '').replace('.png', '')
                        
                        # Position for caption
                        pdf.set_x(pdf.l_margin + j * col_width)
                        pdf.set_font('Arial', '', 7)  # Reduced font size from 8 to 7
                        
                        # Check if line crosses contours to change text color
                        if img_name in crossing_lines:
                            pdf.set_text_color(194, 0, 0)  # Red for lines crossing contours
                        else:
                            pdf.set_text_color(0, 0, 0)  # Black for others
                            
                        pdf.cell(col_width, 5, f"Line: {img_name}", 0, 0, 'C')
                
                # Reset text color
                pdf.set_text_color(0, 0, 0)
                
                # Advance to next row
                pdf.ln(5)
            
            # Add total lines analyzed
            pdf.ln(3)
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, f'Total lines analyzed: {len(analysis_images)}', 0, 1, 'C')
                
        else:
            pdf.set_font('Arial', '', 11)
            pdf.cell(0, 10, 'No line analysis images found', 0, 1)
    else:
        pdf.set_font('Arial', '', 11)
        pdf.cell(0, 10, 'Line analysis directory not found', 0, 1)

def load_crossing_lines_from_csv(csv_path: str) -> List[str]:
    """Loads line names that cross contours from analysis CSV.
    
    Args:
        csv_path: Path to the CSV file with analysis results
        
    Returns:
        List of line names that cross contours
    """
    crossing_lines = set()
    if os.path.exists(csv_path):
        try:
            csv_df = pd.read_csv(csv_path)
            if "Crosses contours" in csv_df.columns:
                crossing_df = csv_df[csv_df["Crosses contours"] == "Yes"]
                crossing_lines = set(crossing_df["Line name"].tolist())
        except:
            pass  # If error, continue without this information
    return crossing_lines
