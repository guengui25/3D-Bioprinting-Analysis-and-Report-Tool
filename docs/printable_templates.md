# Printable Templates for 3D Print Photography

This document explains how to use the printable_template.py script to generate standardized photography templates for capturing images of 3D printed structures.

Location: [`aux/template_generation/printable_template.py`](../aux/template_generation/printable_template.py)

## Overview

The template generation script creates printable PDF templates that serve as photography aids when taking photographs of 3D printed objects for the tools. These templates include reference elements for proper scaling, positioning, and distortion correction during image analysis.

## Features

The generated templates include:
- **50mm Blue Ruler**: Helps detect scale and provides measurement reference
- **Red Square with Circle**: Additional reference point for scale and distortion detection
- **Central Area**: Large rectangle with circle overlay where 3D prints should be placed
  - Available in white (default) or black background for different material contrast needs
- **Reference Card Outline**: Standard credit/ID card dimensions (85.6 x 53.98 mm) for printing scale reference

## Usage

### Generating Templates

1. Ensure you have the ReportLab library installed (included in requirements.txt): 
   ```
   pip install reportlab
   ```

2. Run the script:
   ```
   python printable_template.py
   ```

3. Two PDF files will be generated:
   - `printable_template_white.pdf`: Template with white center background
   - `printable_template_black.pdf`: Template with black center background (better for high-contrast materials)

### Using the Templates

1. Print the appropriate template (white or black) on A4 paper
2. Place a standard credit card or ID card in the outlined rectangle in the bottom right corner to ensure correct scaling of the print (remove the card before taking the photo)
3. Place your 3D printed object in the center of the template (within the large circle)
4. Take a photograph from directly above, ensuring the entire template is visible and avoiding any shadows or reflections
5. Keep the photo as centered and straight as possible, as instructed on the template
6. Use this photograph with the 3D Bioprinting Analysis tool for accurate measurements and analysis of your 3D printed object

## Note

The reference card outline is specifically designed for a standard credit card or ID card (85.6 x 53.98 mm). 

For best results, ensure good lighting and avoid shadows that might interfere with the automatic detection of template elements.