from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A4

def draw_ruler(c, x, y, length_mm, margin_box=6 * mm):
    """Draws a horizontal ruler with the specified length.
    
    Args:
        c: Canvas object to draw on
        x: X-coordinate to start drawing
        y: Y-coordinate to start drawing
        length_mm: Length of the ruler in millimeters
        margin_box: Margin for the ruler box in mm
        
    Returns:
        None
    """
    c.setStrokeColorRGB(0, 0, 1)  # blue
    c.setFillColorRGB(0, 0, 1)

    c.line(x, y, x + length_mm * mm, y)  # base line
    for i in range(length_mm + 1):  # for each millimeter
        x_pos = x + i * mm
        tick_length = 5 * mm if i % 10 == 0 else 3 * mm
        c.line(x_pos, y, x_pos, y + tick_length)
        if i % 10 == 0:
            c.setFont("Helvetica", 6)
            c.drawString(x_pos - 2 * mm, y + tick_length + 4 * mm, f"{i} mm")
    
    # Ruler border
    border_x = x - margin_box
    border_y = y - 4 * mm
    border_width = length_mm * mm + 2 * margin_box
    border_height = 5 * mm + 4 * mm + margin_box + 4 * mm
    c.rect(border_x, border_y, border_width, border_height)

    # Set color back to black for other elements
    c.setStrokeColorRGB(0, 0, 0)
    c.setFillColorRGB(0, 0, 0)

def draw_red_rectangle_with_circle(c, x, y, width_mm, height_mm, circle_diameter_mm):
    """Draws a filled red rectangle with a centered hollow circle inside.
    
    Args:
        c: Canvas object to draw on
        x: X-coordinate of the rectangle
        y: Y-coordinate of the rectangle
        width_mm: Width of the rectangle in millimeters
        height_mm: Height of the rectangle in millimeters
        circle_diameter_mm: Diameter of the circle in millimeters
        
    Returns:
        None
    """
    c.setFillColorRGB(1, 0, 0)  # Red
    c.rect(x, y, width_mm * mm, height_mm * mm, fill=1, stroke=0)  # red square
    cx = x + (width_mm * mm) / 2
    cy = y + (height_mm * mm) / 2
    radius = (circle_diameter_mm * mm) / 2  # circle radius
    c.setFillColorRGB(0, 0, 0)  # Set back to black for the circle
    c.circle(cx, cy, radius, stroke=1, fill=0)

def draw_center_rectangle_with_circle(c, center_x, center_y, rect_width_mm, rect_height_mm, circle_diameter_mm):
    """Draws a rectangle centered at specified coordinates with a circle centered inside.
    
    Args:
        c: Canvas object to draw on
        center_x: X-coordinate of the center
        center_y: Y-coordinate of the center
        rect_width_mm: Width of the rectangle in millimeters
        rect_height_mm: Height of the rectangle in millimeters
        circle_diameter_mm: Diameter of the circle in millimeters
        
    Returns:
        None
    """
    x_rect = center_x - (rect_width_mm * mm) / 2
    y_rect = center_y - (rect_height_mm * mm) / 2
    c.rect(x_rect, y_rect, rect_width_mm * mm, rect_height_mm * mm, stroke=1, fill=0)  # rectangle
    c.circle(center_x, center_y, (circle_diameter_mm * mm) / 2, stroke=1, fill=0)  # circle

def draw_center_rectangle_with_filled_circle(c, center_x, center_y, rect_width_mm, rect_height_mm, circle_diameter_mm):
    """Draws a filled black rectangle centered at specified coordinates with a filled black circle centered inside.
    
    Args:
        c: Canvas object to draw on
        center_x: X-coordinate of the center
        center_y: Y-coordinate of the center
        rect_width_mm: Width of the rectangle in millimeters
        rect_height_mm: Height of the rectangle in millimeters
        circle_diameter_mm: Diameter of the circle in millimeters
        
    Returns:
        None
    """
    x_rect = center_x - (rect_width_mm * mm) / 2
    y_rect = center_y - (rect_height_mm * mm) / 2
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeColorRGB(0.5, 0.5, 0.5)
    c.rect(x_rect, y_rect, rect_width_mm * mm, rect_height_mm * mm, stroke=1, fill=1)
    c.circle(center_x, center_y, (circle_diameter_mm * mm) / 2, stroke=1, fill=1)

def draw_card(c, x, y):
    """Draws a reference card with standard credit card dimensions.
    
    Args:
        c: Canvas object to draw on
        x: X-coordinate of the card
        y: Y-coordinate of the card
        
    Returns:
        None
    """
    card_width = 85.6 * mm
    card_height = 53.98 * mm
    c.rect(x, y, card_width, card_height)  # reference rectangle

def create_pdf():
    """Creates a PDF template in A4 vertical format with various reference elements.
    
    The layout includes:
    - A ruler and red square with circle at the top
    - A central rectangle with overlapping circle
    - A reference card in the bottom right corner
    - Instructional text
    
    Args:
        None
        
    Returns:
        None
    """
    pagesize = A4  # Vertical
    c = canvas.Canvas("printable_template_white.pdf", pagesize=pagesize)
    page_width, page_height = pagesize

    # --- Define central parameters ---
    center_x = page_width / 2
    center_y = page_height / 2

    # --- Size of central figures ---
    center_circle_diameter = 121  # mm
    center_rect_width = 127.80  # mm
    center_rect_height = 85.50   # mm

    # --- Ruler size ---
    rule_length = 50  # mm
    margin_box = 6 * mm  # margin for ruler box
    rule_total_width = rule_length * mm + 2 * margin_box  # total ruler width

    # --- Red square size ---
    red_width = 20  # mm

    # --- Top group: Ruler and red square ---
    gap = 10 * mm  # space between ruler and red square
    extra_offset = 5 * mm  # additional offset to the right for the red square
    group_total_width = rule_total_width + gap + red_width * mm
    group_left = (page_width - group_total_width) / 2  # Center the group horizontally

    # Position the group vertically: 40 mm from the top of the central circle
    top_of_circle = center_y + (center_circle_diameter * mm) / 2
    top_common = top_of_circle + 40 * mm

    # --- Position the figures ---  
    rule_x = group_left
    rule_y = top_common - 15 * mm  # ruler 15 mm above the circle

    # For the red square, with height 20 mm, we want its top to be at top_common
    red_x = group_left + rule_total_width + gap + extra_offset  # red square 5 mm to the right of the ruler
    red_y = top_common - 20 * mm  # red square 20 mm above the circle

    # Draw the ruler (in blue)
    draw_ruler(c, rule_x, rule_y, rule_length, margin_box=6 * mm)

    # Draw the red square with inner circle (circle with 10 mm diameter)
    draw_red_rectangle_with_circle(c, red_x, red_y, red_width, red_width, 10)

    # --- Central element: Horizontal rectangle with overlapping circle ---
    draw_center_rectangle_with_circle(c, center_x, center_y, center_rect_width, center_rect_height, center_circle_diameter)

    # --- Warning text below the large circle ---
    warning_text = "Take the photo as centered and straight as possible"
    # Position the text centered horizontally below the circle, with a 5 mm margin
    warning_x = center_x
    # The bottom edge of the circle is: center_y - (center_circle_diameter*mm)/2
    warning_y = center_y - (center_circle_diameter * mm) / 2 - 15 * mm
    c.setFont("Helvetica", 16)  # Large font
    c.drawCentredString(warning_x, warning_y, warning_text)

    # --- Reference card in the bottom right corner, as close as possible to the bottom edge ---
    card_margin = 5 * mm
    card_width = 85.6 * mm
    card_height = 53.98 * mm
    card_x = page_width - card_width - card_margin
    card_y = card_margin
    draw_card(c, card_x, card_y)

    # Reference card legend, shifted 5 mm to the right and positioned just above the card
    legend_text = "Reference Card (85.6 x 53.98 mm)"
    legend_x = card_x + 5 * mm
    legend_y = card_y + card_height + 5 * mm
    c.setFont("Helvetica", 8)
    c.drawString(legend_x, legend_y, legend_text)

    c.showPage()
    c.save()
    print("PDF 'printable_template_white.pdf' created successfully.")

def create_pdf_black_center():
    """Creates a PDF template similar to the original but with the central rectangle and circle filled in black.
    
    Args:
        None
        
    Returns:
        None
    """
    pagesize = A4  # Vertical
    c = canvas.Canvas("printable_template_black.pdf", pagesize=pagesize)
    page_width, page_height = pagesize

    # --- Define central parameters ---
    center_x = page_width / 2
    center_y = page_height / 2

    # --- Size of central figures ---
    center_circle_diameter = 121  # mm
    center_rect_width = 127.80  # mm
    center_rect_height = 85.50   # mm

    # --- Ruler size ---
    rule_length = 50  # mm
    margin_box = 6 * mm  # margin for ruler box
    rule_total_width = rule_length * mm + 2 * margin_box  # total ruler width

    # --- Red square size ---
    red_width = 20  # mm

    # --- Top group: Ruler and red square ---
    gap = 10 * mm  # space between ruler and red square
    extra_offset = 5 * mm  # additional offset to the right for the red square
    group_total_width = rule_total_width + gap + red_width * mm
    group_left = (page_width - group_total_width) / 2  # Center the group horizontally

    # Position the group vertically: 40 mm from the top of the central circle
    top_of_circle = center_y + (center_circle_diameter * mm) / 2
    top_common = top_of_circle + 40 * mm

    # --- Position the figures ---  
    rule_x = group_left
    rule_y = top_common - 15 * mm  # ruler 15 mm above the circle

    # For the red square, with height 20 mm, we want its top to be at top_common
    red_x = group_left + rule_total_width + gap + extra_offset  # red square 5 mm to the right of the ruler
    red_y = top_common - 20 * mm  # red square 20 mm above the circle

    # Draw the ruler (in blue)
    draw_ruler(c, rule_x, rule_y, rule_length, margin_box=6 * mm)

    # Draw the red square with inner circle (circle with 10 mm diameter)
    draw_red_rectangle_with_circle(c, red_x, red_y, red_width, red_width, 10)

    # --- Central element: Horizontal rectangle with overlapping circle, both filled in black ---
    draw_center_rectangle_with_filled_circle(c, center_x, center_y, center_rect_width, center_rect_height, center_circle_diameter)

    # --- Warning text below the large circle ---
    warning_text = "Take the photo as centered and straight as possible"
    warning_x = center_x
    warning_y = center_y - (center_circle_diameter * mm) / 2 - 15 * mm
    c.setFont("Helvetica", 16)
    c.drawCentredString(warning_x, warning_y, warning_text)

    # --- Reference card in the bottom right corner ---
    card_margin = 5 * mm
    card_width = 85.6 * mm
    card_height = 53.98 * mm
    card_x = page_width - card_width - card_margin
    card_y = card_margin
    draw_card(c, card_x, card_y)

    # Reference card legend
    legend_text = "Reference Card (85.6 x 53.98 mm)"
    legend_x = card_x + 5 * mm
    legend_y = card_y + card_height + 5 * mm
    c.setFont("Helvetica", 8)
    c.drawString(legend_x, legend_y, legend_text)

    c.showPage()
    c.save()
    print("PDF 'printable_template_black.pdf' created successfully.")

if __name__ == "__main__":
    create_pdf()
    create_pdf_black_center()