import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import sys
import queue
from datetime import datetime
import logging

# Import functions from main modules
from app.main_gcode import parse_gcode_file, plot_timelapse_highlight_current, limit_layers
from app.main_mask import main as process_mask

class RedirectText:
    """
    Class to redirect stdout to a tkinter Text widget with optimized performance
    using a queue to avoid GUI freezing
    """
    def __init__(self, text_widget, app_instance):
        self.text_widget = text_widget
        self.app_instance = app_instance
        self.queue = queue.Queue()
        self.update_interval = 100  # milliseconds
        self.updating = False
        
    def write(self, string):
        self.queue.put(string)
        if not self.updating:
            self.updating = True
            self.app_instance.after(0, self.update_text_widget)
    
    def update_text_widget(self):
        """Update the text widget with all queued text"""
        try:
            # Get all available text from queue
            text = ""
            while not self.queue.empty():
                text += self.queue.get_nowait()
            
            if text:
                self.text_widget.insert(tk.END, text)
                self.text_widget.see(tk.END)
                self.app_instance.update_idletasks()
            
            # Schedule next update if there's more text or reschedule check
            if not self.queue.empty():
                self.app_instance.after(10, self.update_text_widget)
            else:
                self.updating = False
                self.app_instance.after(self.update_interval, self.check_queue)
        except Exception as e:
            print(f"Error updating text widget: {e}")
            self.updating = False
    
    def check_queue(self):
        """Check if there's text in the queue and update if needed"""
        if not self.queue.empty() and not self.updating:
            self.updating = True
            self.update_text_widget()
        elif not self.updating:
            self.app_instance.after(self.update_interval, self.check_queue)

    def flush(self):
        pass

class ProcessingApp(tk.Tk):
    """Main application class for G-Code and Image Processing GUI"""
    
    def __init__(self):
        super().__init__()
        
        # Configure main window
        self.title("3D Bioprinting Analysis and Report Tool")
        self.geometry("900x700")
        self.configure(padx=10, pady=10)
        
        # Set up logging
        self.setup_logging()
        
        # Create tab control
        self.tab_control = ttk.Notebook(self)
        
        # Create tabs
        self.gcode_tab = ttk.Frame(self.tab_control)
        self.image_tab = ttk.Frame(self.tab_control)
        
        # Add tabs to notebook
        self.tab_control.add(self.gcode_tab, text="G-Code Processing")
        self.tab_control.add(self.image_tab, text="Image Processing")
        self.tab_control.pack(expand=1, fill="both")
        
        # Set up G-Code processing tab
        self.setup_gcode_tab()

        # Image tab variables for width samples CSV export
        self.export_samples_csv = tk.BooleanVar(value=True)
        self.num_samples_var = tk.IntVar(value=20)

        # Set up Image processing tab
        self.setup_image_tab()

        # Setup common components
        self.setup_common_components()

        # Initialize state variables
        self.gcode_processing_thread = None
        self.image_processing_thread = None

        # Add periodic UI update
        self.update_interval = 100  # milliseconds
        self.after(self.update_interval, self.periodic_ui_update)

    def setup_logging(self):
        """Configure logging to file and console"""
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("ProcessingApp")
        self.logger.info("Application started")

    def setup_gcode_tab(self):
        """Set up the G-Code processing tab"""
        frame = self.gcode_tab
        
        # File selection section
        file_frame = ttk.LabelFrame(frame, text="G-Code File")
        file_frame.pack(fill="x", padx=5, pady=5)
        
        self.gcode_file_path = tk.StringVar()
        ttk.Label(file_frame, text="G-Code File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(file_frame, textvariable=self.gcode_file_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_gcode_file).grid(row=0, column=2, padx=5, pady=5)
        
        self.output_folder_path = tk.StringVar(value=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output/gcode"))
        ttk.Label(file_frame, text="Output Folder:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(file_frame, textvariable=self.output_folder_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_output_folder).grid(row=1, column=2, padx=5, pady=5)
        
        # Options section
        options_frame = ttk.LabelFrame(frame, text="Processing Options")
        options_frame.pack(fill="x", padx=5, pady=5)
        
        self.skip_perimeter = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Skip Perimeter", variable=self.skip_perimeter).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.perimeter_percentage = tk.IntVar(value=30)
        ttk.Label(options_frame, text="Perimeter %:").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        ttk.Spinbox(options_frame, from_=0, to=100, textvariable=self.perimeter_percentage, width=5).grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        self.limit_layers_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Limit Layers", variable=self.limit_layers_var).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.max_layers = tk.IntVar(value=2)
        ttk.Label(options_frame, text="Max Layers:").grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Spinbox(options_frame, from_=1, to=100, textvariable=self.max_layers, width=5).grid(row=1, column=2, padx=5, pady=5, sticky="w")
        
        # Process button
        ttk.Button(frame, text="Process G-Code", command=self.process_gcode).pack(pady=10)
        
        # Console output
        console_frame = ttk.LabelFrame(frame, text="Console Output")
        console_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add clear button for console
        clear_btn = ttk.Button(console_frame, text="Clear Output", command=self.clear_gcode_console)
        clear_btn.pack(anchor="ne", padx=5, pady=2)
        
        self.gcode_console = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, height=10)
        self.gcode_console.pack(fill="both", expand=True, padx=5, pady=5)

    def setup_image_tab(self):
        """Set up the Image processing tab"""
        frame = self.image_tab
        
        # File selection section
        file_frame = ttk.LabelFrame(frame, text="Image and G-Code Images Directory")
        file_frame.pack(fill="x", padx=5, pady=5)
        
        self.image_file_path = tk.StringVar()
        ttk.Label(file_frame, text="Image File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(file_frame, textvariable=self.image_file_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_image_file).grid(row=0, column=2, padx=5, pady=5)
        
        self.gcode_dir_path = tk.StringVar()
        ttk.Label(file_frame, text="G-Code Images Directory:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(file_frame, textvariable=self.gcode_dir_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_gcode_dir).grid(row=1, column=2, padx=5, pady=5)
        
        # Output directory selection
        self.image_output_dir = tk.StringVar(value=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"))
        ttk.Label(file_frame, text="Output Directory:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(file_frame, textvariable=self.image_output_dir, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_image_output_dir).grid(row=2, column=2, padx=5, pady=5)
        
        # Options section
        options_frame = ttk.LabelFrame(frame, text="Processing Options")
        options_frame.pack(fill="x", padx=5, pady=5)
        
        self.save_analysis_images = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Save Analysis Images", variable=self.save_analysis_images).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.generate_gif = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Generate GIF", variable=self.generate_gif).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.generate_report = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Generate PDF Report", variable=self.generate_report).grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Options for width samples CSV
        ttk.Checkbutton(options_frame, text="Export width samples CSV", variable=self.export_samples_csv) \
            .grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Label(options_frame, text="Number of samples:") \
            .grid(row=1, column=1, padx=5, pady=5, sticky="w")
        ttk.Spinbox(options_frame, from_=1, to=100, textvariable=self.num_samples_var, width=5) \
            .grid(row=1, column=2, padx=5, pady=5, sticky="w")
        
        # Process button
        ttk.Button(frame, text="Process Image", command=self.process_image).pack(pady=10)
        
        # Console output
        console_frame = ttk.LabelFrame(frame, text="Console Output")
        console_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add clear button for console
        clear_btn = ttk.Button(console_frame, text="Clear Output", command=self.clear_image_console)
        clear_btn.pack(anchor="ne", padx=5, pady=2)
        
        self.image_console = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, height=10)
        self.image_console.pack(fill="both", expand=True, padx=5, pady=5)

    def setup_common_components(self):
        """Set up components common to both tabs"""
        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", side="bottom", padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side="left")
        
        # Add a progress label that can show "Processing..." when needed
        self.progress_label = ttk.Label(status_frame, text="")
        self.progress_label.pack(side="left", padx=10)
        
        ttk.Button(status_frame, text="Exit", command=self.quit).pack(side="right")

    def browse_gcode_file(self):
        """Open file dialog to select G-Code file"""
        file_path = filedialog.askopenfilename(
            title="Select G-Code File",
            filetypes=[("G-Code Files", "*.gcode"), ("All Files", "*.*")]
        )
        if file_path:
            self.gcode_file_path.set(file_path)
            self.logger.info(f"G-Code file selected: {file_path}")

    def browse_output_folder(self):
        """Open directory dialog to select output folder"""
        folder_path = filedialog.askdirectory(
            title="Select Output Folder"
        )
        if folder_path:
            self.output_folder_path.set(folder_path)
            self.logger.info(f"Output folder selected: {folder_path}")

    def browse_image_file(self):
        """Open file dialog to select image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image Files", "*.png *.jpg *.jpeg *.tiff *.bmp"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.image_file_path.set(file_path)
            self.logger.info(f"Image file selected: {file_path}")

    def browse_gcode_dir(self):
        """Open directory dialog to select G-Code directory"""
        dir_path = filedialog.askdirectory(
            title="Select G-Code Directory"
        )
        if dir_path:
            self.gcode_dir_path.set(dir_path)
            self.logger.info(f"G-Code directory selected: {dir_path}")

    def browse_image_output_dir(self):
        """Open directory dialog to select image output directory"""
        dir_path = filedialog.askdirectory(
            title="Select Image Output Directory"
        )
        if dir_path:
            self.image_output_dir.set(dir_path)
            self.logger.info(f"Image output directory selected: {dir_path}")

    def process_gcode(self):
        """Process the selected G-Code file"""
        if not self.gcode_file_path.get():
            messagebox.showerror("Error", "Please select a G-Code file")
            return
        
        if not os.path.exists(self.gcode_file_path.get()):
            messagebox.showerror("Error", "Selected G-Code file does not exist")
            return
            
        # Make sure output directory exists
        os.makedirs(self.output_folder_path.get(), exist_ok=True)
        
        # Clear console
        self.gcode_console.delete(1.0, tk.END)
        
        # Redirect stdout to the console with improved redirection
        gcode_redirect = RedirectText(self.gcode_console, self)
        original_stdout = sys.stdout
        sys.stdout = gcode_redirect
        
        # Start processing in a separate thread to avoid freezing the UI
        self.status_var.set("Processing G-Code...")
        
        # Update status indicator
        self.update_status_indicator(True)
        
        self.gcode_processing_thread = threading.Thread(
            target=self._process_gcode_thread,
            args=(original_stdout,),
            daemon=True
        )
        self.gcode_processing_thread.start()
    
    def _process_gcode_thread(self, original_stdout):
        """Thread function for G-Code processing"""
        try:
            gcode_file = self.gcode_file_path.get()
            output_folder = self.output_folder_path.get()
            skip_perimeter = self.skip_perimeter.get()
            perimeter_percentage = self.perimeter_percentage.get()
            
            # Parse G-Code file
            layers, nozzle_diameter = parse_gcode_file(
                gcode_file, 
                skip_perimeter=skip_perimeter, 
                perimeter_percentage=perimeter_percentage
            )
            
            # Limit layers if requested
            max_layers = None
            if self.limit_layers_var.get():
                max_layers = self.max_layers.get()
            
            layers = limit_layers(layers, max_layers)
            
            # Generate images
            plot_timelapse_highlight_current(
                layers, 
                gcode_file, 
                output_folder=output_folder, 
                nozzle_diameter=nozzle_diameter
            )
            
            # Output the created directory - will be needed for image processing
            base_filename = os.path.basename(gcode_file)
            base_name, _ = os.path.splitext(base_filename)
            gcode_dir = os.path.join(output_folder, f"{base_name}_highlight")
            
            # Update UI from the main thread
            self.after(0, lambda: self._on_gcode_complete(gcode_dir))
            
        except Exception as e:
            self.logger.exception("Error in G-Code processing")
            # Update UI from the main thread
            self.after(0, lambda: self._on_processing_error(str(e)))
            
        finally:
            # Restore stdout in the main thread
            self.after(0, lambda: setattr(sys, 'stdout', original_stdout))
            # Update status indicator
            self.after(0, lambda: self.update_status_indicator(False))
    
    def _on_gcode_complete(self, gcode_dir):
        """Called when G-Code processing is complete"""
        self.status_var.set("G-Code Processing Complete")
        messagebox.showinfo("Success", f"G-Code processing complete.\nOutput directory: {gcode_dir}")
        
        # Auto-set the G-Code directory in the image tab
        self.gcode_dir_path.set(gcode_dir)
        
        # Switch to the image processing tab
        self.tab_control.select(self.image_tab)

    def process_image(self):
        """Process the selected image with the G-Code directory"""
        if not self.image_file_path.get():
            messagebox.showerror("Error", "Please select an image file")
            return
        
        if not os.path.exists(self.image_file_path.get()):
            messagebox.showerror("Error", "Selected image file does not exist")
            return
            
        if not self.gcode_dir_path.get():
            messagebox.showerror("Error", "Please select a G-Code directory")
            return
            
        if not os.path.exists(self.gcode_dir_path.get()):
            messagebox.showerror("Error", "Selected G-Code directory does not exist")
            return
        
        # Clear console
        self.image_console.delete(1.0, tk.END)
        
        # Redirect stdout to the console with improved redirection
        image_redirect = RedirectText(self.image_console, self)
        original_stdout = sys.stdout
        sys.stdout = image_redirect
        
        # Start processing in a separate thread to avoid freezing the UI
        self.status_var.set("Processing Image...")
        
        # Update status indicator
        self.update_status_indicator(True)
        
        self.image_processing_thread = threading.Thread(
            target=self._process_image_thread,
            args=(original_stdout,),
            daemon=True
        )
        self.image_processing_thread.start()
    
    def _process_image_thread(self, original_stdout):
        """Thread function for Image processing"""
        try:
            image_path = self.image_file_path.get()
            gcode_dir_path = self.gcode_dir_path.get()
            output_dir = self.image_output_dir.get()
            
            # Process the image
            results = process_mask(
                image_path=image_path,
                gcode_folder_path=gcode_dir_path,
                save_analysis_images=self.save_analysis_images.get(),
                generate_gif=self.generate_gif.get(),
                generate_report_pdf=self.generate_report.get(),
                base_output_dir=output_dir,
                export_samples_csv=self.export_samples_csv.get(),
                num_samples=self.num_samples_var.get()
            )
            
            # Get output directory path from the results if available
            result_output_dir = None
            if results and isinstance(results, tuple) and len(results) > 3:
                analysis_results = results[3]
                if isinstance(analysis_results, dict) and 'output_dir' in analysis_results:
                    result_output_dir = analysis_results['output_dir']
            
            # Update UI from the main thread
            self.after(0, lambda: self._on_image_complete(result_output_dir))
            
        except Exception as e:
            self.logger.exception("Error in image processing")
            # Update UI from the main thread
            self.after(0, lambda: self._on_processing_error(str(e)))
            
        finally:
            # Restore stdout in the main thread
            self.after(0, lambda: setattr(sys, 'stdout', original_stdout))
            # Update status indicator
            self.after(0, lambda: self.update_status_indicator(False))
    
    def _on_image_complete(self, output_dir):
        """Called when image processing is complete"""
        self.status_var.set("Image Processing Complete")
        
        if output_dir and os.path.exists(output_dir):
            messagebox.showinfo("Success", f"Image processing complete.\nOutput directory: {output_dir}")
        else:
            messagebox.showinfo("Success", "Image processing complete.")
    
    def _on_processing_error(self, error_msg):
        """Called when an error occurs during processing"""
        self.status_var.set("Error")
        messagebox.showerror("Processing Error", f"An error occurred during processing:\n\n{error_msg}")

    def update_status_indicator(self, is_processing):
        """Update status indicators to show processing state"""
        if is_processing:
            self.status_var.set("Processing... Please wait")
            self.progress_label.config(text="Processing...")
            # Force an update of the UI
            self.update_idletasks()
        else:
            self.status_var.set("Ready")
            self.progress_label.config(text="")
            self.update_idletasks()

    def periodic_ui_update(self):
        """Periodically update the UI to ensure responsiveness"""
        try:
            self.update_idletasks()
        except Exception as e:
            self.logger.error(f"Error in periodic UI update: {e}")
        finally:
            # Schedule the next update
            self.after(self.update_interval, self.periodic_ui_update)

    def clear_gcode_console(self):
        """Clear the G-Code console output"""
        self.gcode_console.delete(1.0, tk.END)
        
    def clear_image_console(self):
        """Clear the image console output"""
        self.image_console.delete(1.0, tk.END)


if __name__ == "__main__":
    app = ProcessingApp()
    app.mainloop()
