import matplotlib
matplotlib.use('Agg') # Force matplotlib to use the non-interactive Agg backend which is thread-safe

import matplotlib.pyplot as plt
import os
import re
import math

def parse_coordinate(line, coord):
    """Searches for coordinate values in G-Code lines.
    
    Args:
        line: The G-Code line to search in
        coord: The coordinate letter to look for (e.g., 'X', 'Y')
        
    Returns:
        Float value of the coordinate if found, None otherwise
    """
    if coord not in line:
        return None
    match = re.search(rf"{coord}([-]?\d+(\.\d+)?)", line)
    if match:
        return float(match.group(1))
    return None

def extract_nozzle_diameter(line):
    """Extracts nozzle diameter from G-code comment lines.
    
    Args:
        line: A line from the G-code file to parse
        
    Returns:
        Float value of the nozzle diameter if found, None otherwise
    """
    if not line.startswith('; nozzle_diameter'):
        return None
        
    parts = line.split('=')
    if len(parts) != 2:
        return None
        
    values = parts[1].strip().split(',')
    if not values or not values[0]:
        return None
        
    try:
        print (f"Extracted nozzle diameter: {values[0]}")
        return float(values[0])
    except ValueError:
        return None

def process_movement_command(line, current_pos, relative_extrusion, temp_layers):
    """Processes G0/G1 movement commands to extract extrusion segments.
    
    Args:
        line: The G-code line containing movement commands
        current_pos: Tuple of current (x, y, z, e) position
        relative_extrusion: Whether extrusion is relative or absolute
        temp_layers: Dictionary of current layers and segments
        
    Returns:
        Tuple containing updated current position and updated layers dictionary
    """
    current_x, current_y, current_z, current_e = current_pos
    
    new_x = parse_coordinate(line, 'X')
    new_y = parse_coordinate(line, 'Y')
    new_z = parse_coordinate(line, 'Z')
    new_e = parse_coordinate(line, 'E')

    # Update Z-layer if changed
    if new_z is not None and new_z != current_z:
        current_z = new_z

    # Update X position
    if new_x is not None:
        x_prev, current_x = current_x, new_x
    else:
        x_prev = current_x

    # Update Y position
    if new_y is not None:
        y_prev, current_y = current_y, new_y
    else:
        y_prev = current_y

    # If there's extrusion, save the segment
    if new_e is not None:
        if (relative_extrusion and new_e > 0) or (not relative_extrusion and new_e > current_e):
            if current_z not in temp_layers:
                temp_layers[current_z] = []
            temp_layers[current_z].append(((x_prev, y_prev), (current_x, current_y)))
        if not relative_extrusion:
            current_e = new_e
            
    return (current_x, current_y, current_z, current_e), temp_layers

def parse_gcode_file(filename, skip_perimeter=True, perimeter_percentage=30):
    """Reads a G-Code file from Prusa Slicer and extracts extrusion segments by layer.
    
    Args:
        filename: Path to the G-code file
        skip_perimeter: If True, skips the initial outer perimeter
        perimeter_percentage: Percentage of initial segments to consider as perimeter
        
    Returns:
        Tuple containing dictionary of layers and the nozzle diameter value
    """
    relative_extrusion = False
    current_pos = (0.0, 0.0, 0.0, 0.0)  # (x, y, z, e)
    nozzle_diameter = 0.840  # Default value if not found in the file
    
    # Dictionary to store segments by layer
    temp_layers = {}
    
    # First pass: collect all segments and organize them by layer
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Extract nozzle diameter from comments if available
            detected_diameter = extract_nozzle_diameter(line)
            if detected_diameter:
                nozzle_diameter = detected_diameter
                print(f"Detected nozzle diameter: {nozzle_diameter} mm")
                continue
                            
            if line.startswith('M83'):
                relative_extrusion = True
                continue
                
            # Skip comment lines
            if line.startswith(';'):
                continue

            if line.startswith('G0') or line.startswith('G1'):
                current_pos, temp_layers = process_movement_command(
                    line, current_pos, relative_extrusion, temp_layers
                )
    
    # If we don't need to skip the perimeter, return all layers as is
    if not skip_perimeter:
        return temp_layers, nozzle_diameter
    
    return filter_perimeter(temp_layers, perimeter_percentage), nozzle_diameter

def filter_perimeter(temp_layers, perimeter_percentage):
    """Processes layers to remove the outer perimeter segments.
    
    Args:
        temp_layers: Dictionary of layers containing all segments
        perimeter_percentage: Percentage of segments to consider as perimeter for fallback
        
    Returns:
        Dictionary of layers with perimeter segments removed from the first layer
    """
    layers = {}
    
    # Sort layers by Z height
    sorted_z_keys = sorted(temp_layers.keys())
    
    # Process the first layer to remove the perimeter
    if sorted_z_keys:
        first_z = sorted_z_keys[0]
        first_layer_segments = temp_layers[first_z]
        
        if first_layer_segments:
            print(f"Detected {len(first_layer_segments)} segments in the first layer")
            
            # Auto-detect perimeter
            perimeter_end_idx = detect_perimeter_end(first_layer_segments)
            
            if perimeter_end_idx > 0:
                print(f"Outer perimeter detected: removing first {perimeter_end_idx} segments")
                non_perimeter_segments = first_layer_segments[perimeter_end_idx:]
                if non_perimeter_segments:
                    layers[first_z] = non_perimeter_segments
                    print(f"First layer after filtering: {len(non_perimeter_segments)} segments")
            else:
                # Fallback to percentage if auto-detection fails
                perimeter_count = max(int(len(first_layer_segments) * perimeter_percentage / 100), 1)
                print(f"Auto-detection failed. Using percentage: skipping first {perimeter_count} segments ({perimeter_percentage}%)")
                non_perimeter_segments = first_layer_segments[perimeter_count:]
                if non_perimeter_segments:
                    layers[first_z] = non_perimeter_segments
                    print(f"First layer after filtering: {len(non_perimeter_segments)} segments")
        
        # Copy all other layers as is
        for z in sorted_z_keys[1:]:
            layers[z] = temp_layers[z]
    
    if not layers:
        print("Warning: No layers or segments found after filtering the perimeter.")
    
    return layers

def detect_perimeter_end(segments):
    """Automatically detects where the outer perimeter ends in the first layer.
    
    Args:
        segments: List of segments from the first layer
        
    Returns:
        Index where the outer perimeter ends, or 0 if it cannot be determined
    """
    if len(segments) < 10:
        return 0  # Too few segments to detect a pattern
    
    # 1. Detect patterns of segments that form a closed loop
    start_point = segments[0][0]
    tolerance = 0.1  # mm tolerance to consider coincident points
    
    # Look for a closed cycle in the first segments
    for i in range(3, min(200, len(segments))):
        end_point = segments[i][1]
        
        # Check if the endpoint is close to the starting point (forming a loop)
        dist = ((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)**0.5
        if dist <= tolerance:
            return i + 1  # Return index after the final segment of the perimeter
    
    # 2. Analyze changes in segment directions
    if len(segments) >= 20:
        # Calculate the average initial direction of the first segments
        directions = []
        for i in range(min(10, len(segments))):
            (x1, y1), (x2, y2) = segments[i]
            if abs(x2 - x1) > 0.001 or abs(y2 - y1) > 0.001:  # Avoid very small segments
                angle = math.atan2(y2 - y1, x2 - x1)
                directions.append(angle)
        
        if directions:
            # Look for an abrupt change in direction, which might indicate the end of the perimeter
            for i in range(10, min(150, len(segments))):
                (x1, y1), (x2, y2) = segments[i]
                if abs(x2 - x1) > 0.001 or abs(y2 - y1) > 0.001:
                    current_angle = math.atan2(y2 - y1, x2 - x1)
                    
                    # Check if there's a significant change in direction
                    angle_changes = [min(abs(current_angle - prev), 2*math.pi - abs(current_angle - prev)) 
                                     for prev in directions[-5:]]
                    
                    # Also check for changes in segment length
                    seg_length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                    prev_length = ((segments[i-1][1][0] - segments[i-1][0][0])**2 + 
                                  (segments[i-1][1][1] - segments[i-1][0][1])**2)**0.5
                    
                    # If there are significant changes in both direction and length
                    if (max(angle_changes) > math.pi/4 and 
                        (seg_length > 2 * prev_length or seg_length < 0.5 * prev_length)):
                        return i
    
    # 3. Look for patterns of rectangles or horizontal/vertical lines that often
    # indicate the start of infill after the outer perimeters
    horizontal_vertical_count = 0
    transition_idx = 0
    
    for i in range(1, min(150, len(segments))):
        (x1, y1), (x2, y2) = segments[i]
        
        # Check if the segment is approximately horizontal or vertical
        is_horizontal = abs(y2 - y1) < tolerance
        is_vertical = abs(x2 - x1) < tolerance
        
        if is_horizontal or is_vertical:
            horizontal_vertical_count += 1
            
            # If we find several consecutive H/V segments after the first segments
            # we're probably in the interior fill
            if horizontal_vertical_count >= 3 and i > 15:
                transition_idx = i - horizontal_vertical_count + 1
                return transition_idx
        else:
            horizontal_vertical_count = 0
    
    # No clear pattern detected
    return 0

def calculate_plot_bounds(layers):
    """Calculates the boundaries for plotting all segments.
    
    Args:
        layers: Dictionary of layers containing segments
        
    Returns:
        Tuple containing center_x, center_y, radius, and buffer factor for plotting
    """
    all_points = []
    for z, segments in layers.items():
        for seg in segments:
            (x1, y1), (x2, y2) = seg
            all_points.append((x1, y1))
            all_points.append((x2, y2))
    
    if not all_points:
        print("No segments found to draw.")
        return None, None, None, 1.0
    
    xs, ys = zip(*all_points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    # Radius that determines half of the maximum extension (to fix axes)
    r = max((max_x - min_x) / 2, (max_y - min_y) / 2)
    if r == 0:
        r = 1.0
    
    return center_x, center_y, r, 1.1  # Return with buffer factor

def create_output_directory(gcode_filename, output_folder):
    """Creates a directory for output images based on the G-code filename.
    
    Args:
        gcode_filename: Name of the G-code file
        output_folder: Base folder where output should be saved
        
    Returns:
        Path to the created subfolder for outputs
    """
    base_filename = os.path.basename(gcode_filename)
    base_name, _ = os.path.splitext(base_filename)
    
    subfolder_path = os.path.join(output_folder, f"{base_name}_highlight")
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    
    return subfolder_path

def add_scale_bar(ax, scale_length):
    """Adds a scale bar to the bottom-right corner of the plot.
    
    Args:
        ax: Matplotlib Axes object where the scale will be added
        scale_length: Length of the scale (in mm) to be represented
        
    Returns:
        None
    """
    # Get current plot limits (assumed to be centered and in mm)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    r = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]))
    
    margin = 0.05 * r
    rect_height = 0.08 * r  # used to position the line and text
    
    x_end = r - margin
    y_bottom = -r + margin
    line_x_start = x_end - scale_length
    
    # Draw the red line, located at 20% of height from bottom
    line_y = y_bottom + 0.2 * rect_height
    ax.plot([line_x_start, x_end], [line_y, line_y], color='red', linewidth=2)
    
    # Add the scale text centered, located at 60% of height from bottom
    text_y = y_bottom + 0.6 * rect_height
    ax.text(line_x_start + scale_length/2, text_y,
            f"{scale_length} mm", color='red', fontsize=10, ha='center', va='center')

def plot_timelapse_highlight_current(layers, gcode_filename, output_folder="output", nozzle_diameter=0.4):
    """Generates cumulative images showing each extrusion line being printed.
    
    Args:
        layers: Dictionary of Z-heights with associated line segments
        gcode_filename: Name of the G-code file
        output_folder: Folder where images will be saved
        nozzle_diameter: Diameter for line thickness in visualization
        
    Returns:
        None
    """
    # Create output directory
    subfolder_path = create_output_directory(gcode_filename, output_folder)
    
    # Calculate plot boundaries
    center_x, center_y, r, buffer = calculate_plot_bounds(layers)
    if center_x is None:
        return
    
    # Sort layers by Z value
    sorted_layers = sorted(layers.items(), key=lambda item: item[0])
    
    # Accumulate segments as we progress through layers and lines
    cumulative_segments = []
    
    # Global counter to enumerate images in order of appearance
    img_counter = 0
    
    for z, segments in sorted_layers:
        # For each layer, process each segment individually
        for idx, seg in enumerate(segments):
            (x1, y1), (x2, y2) = seg
            # Recenter points in the bounding box
            x1c = x1 - center_x
            y1c = y1 - center_y
            x2c = x2 - center_x
            y2c = y2 - center_y
            
            # Add the current segment to the accumulated list
            cumulative_segments.append((x1c, y1c, x2c, y2c))
            
            # Open figure to draw layers up to the current segment
            plt.figure(figsize=(6, 6), frameon=False)
            plt.xlim(-buffer * r, buffer * r)
            plt.ylim(-buffer * r, buffer * r)
            ax = plt.gca()
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            add_scale_bar(ax, 10)  # Add a 10 mm scale
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Draw all accumulated segments except the last (current) one
            for i, (x1p, y1p, x2p, y2p) in enumerate(cumulative_segments):
                if i < len(cumulative_segments) - 1:  # All previous segments
                    plt.plot([x1p, x2p], [y1p, y2p], linewidth=nozzle_diameter, color='black')
                else:  # Current segment (last in the list)
                    plt.plot([x1p, x2p], [y1p, y2p], linewidth=nozzle_diameter, color='#00FF00')
            
            # Save figure to a file in the subfolder
            file_name = os.path.join(subfolder_path, f"line_{img_counter:06d}_layer_{z:.2f}.png")
            plt.savefig(file_name, dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close()
            img_counter += 1
            if img_counter % 100 == 0:
                print(f"Image {img_counter} saved: {file_name}")
    
    print(f"Total images generated: {img_counter}")

def limit_layers(layers, max_layers):
    """Limits the number of layers to process for optimization.
    
    Args:
        layers: Dictionary of all layers
        max_layers: Maximum number of layers to include
        
    Returns:
        Dictionary containing limited number of layers
    """
    if max_layers is None:
        return layers
        
    sorted_z_keys = sorted(layers.keys())
    if not sorted_z_keys:
        return layers
        
    sorted_z_keys = sorted_z_keys[:min(max_layers, len(sorted_z_keys))]
    return {z: layers[z] for z in sorted_z_keys}

if __name__ == "__main__":
    # Update paths to be one level up from the script location
    gcode_file = "your_gcode_file.gcode"  # Replace with your G-code file if you want to test
    output_folder = "your_output_folder"  # Replace with your output folder if you want to test
    
    # You can adjust the percentage of segments considered as perimeter (default is 30%)
    # A higher value will eliminate more segments from the beginning
    layers, nozzle_diameter = parse_gcode_file(gcode_file, skip_perimeter=True, perimeter_percentage=70)
    
    # Optional: limit the number of layers
    max_layers = 2
    layers = limit_layers(layers, max_layers)
    sorted_z_keys = sorted(layers.keys())
    print(sorted_z_keys)
    
    # Generate images with current line highlighted in green
    plot_timelapse_highlight_current(layers, gcode_file, output_folder=output_folder, nozzle_diameter=nozzle_diameter)