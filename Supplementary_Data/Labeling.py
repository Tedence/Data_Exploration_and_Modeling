import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
base_path = Path(__file__).parent / "Spectrograms"

class SpectrogramLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectrogram Labeler - Tedence")
        self.root.geometry("1400x850")

        # Set the spectrogram directory
        self.spectrograms_dir = base_path
        if not self.spectrograms_dir.exists():
            self.spectrograms_dir = Path(filedialog.askdirectory(
                title="Select Spectrograms Directory"))

        # Initialize variables
        self.image_files = self.get_image_files()
        self.current_image_idx = 0
        self.image = None
        self.photo = None
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.selected_regions = []
        self.current_label = tk.StringVar(value="hypo precursor")

        # Track rectangle being drawn
        self.drawing = False

        # Image metadata
        self.image_metadata = None
        self.scale_factor = 1.0  # For zoom
        self.zoom_levels = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
        self.zoom_idx = self.zoom_levels.index(1.0)  # Start at 100% zoom

        # Load spectrogram bounds
        self.bounds_file = Path("spectrogram_bounds.json")
        self.spectrogram_bounds = self.load_spectrogram_bounds()

        # Variable for coordinate display
        self.coord_text_id = None

        # Load existing labels if available
        self.labels_file = Path("spectrogram_labels.json")
        self.labels_data = self.load_labels()

        # Create UI elements
        self.create_ui()

        # Bind window resize event
        self.root.bind("<Configure>", self.on_window_resize)

        # Load first image
        if self.image_files:
            self.load_image(self.image_files[self.current_image_idx])

            # Auto-fit to window
            self.root.update()
            self.fit_to_window()

    def get_image_files(self):
        """Get all image files from the spectrograms directory"""
        files = []
        if self.spectrograms_dir.exists():
            for file in self.spectrograms_dir.glob("*_spectrogram_with_signal.png"):
                files.append(str(file))
        return sorted(files)

    def load_labels(self):
        """Load existing labels from JSON file"""
        if self.labels_file.exists():
            try:
                with open(self.labels_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                messagebox.showerror("Error", "Labels file is corrupted. Starting fresh.")
                return {}
        return {}

    def save_labels(self):
        """Save labels to JSON file"""
        with open(self.labels_file, 'w') as f:
            json.dump(self.labels_data, f, indent=2)
        messagebox.showinfo("Success", f"Labels saved to {self.labels_file}")

    def create_ui(self):
        """Create the user interface elements"""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, side=tk.TOP, pady=(0, 10))

        # Navigation buttons
        ttk.Button(toolbar, text="Previous", command=self.prev_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Open File...", command=self.open_file_dialog).pack(side=tk.LEFT, padx=5)

        # Current image indicator
        self.image_label = ttk.Label(toolbar, text="Image 0/0")
        self.image_label.pack(side=tk.LEFT, padx=10)

        # Zoom controls
        zoom_frame = ttk.Frame(toolbar)
        zoom_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="-", width=2, command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(zoom_frame, text="100%", width=5)
        self.zoom_label.pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="+", width=2, command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Reset", command=self.reset_zoom).pack(side=tk.LEFT, padx=5)

        # Label selection
        ttk.Label(toolbar, text="Label:").pack(side=tk.LEFT, padx=(20, 5))
        ttk.Radiobutton(toolbar, text="Hypo Precursor", variable=self.current_label,
                      value="hypo precursor").pack(side=tk.LEFT)
        ttk.Radiobutton(toolbar, text="Hyper Precursor", variable=self.current_label,
                      value="hyper precursor").pack(side=tk.LEFT)

        # Save button
        ttk.Button(toolbar, text="Save Labels", command=self.save_labels).pack(side=tk.RIGHT)
        ttk.Button(toolbar, text="Clear Current Image", command=self.clear_current_image).pack(side=tk.RIGHT, padx=5)
        ttk.Button(toolbar, text="Export to CSV", command=self.export_to_csv).pack(side=tk.RIGHT, padx=5)

        # Create canvas for image display
        self.canvas_frame = ttk.Frame(main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas scrollbars
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(fill=tk.X, side=tk.BOTTOM)

        v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(fill=tk.Y, side=tk.RIGHT)

        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        # Canvas events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux scroll down
        self.canvas.bind("<Motion>", self.on_cursor_move)      # Track cursor motion for coordinate display

        # Add panning support with middle mouse button
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan)

        # Add double-click to center the view on that point
        self.canvas.bind("<Double-Button-1>", self.center_view)

        # Bottom status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))

        # Regions display frame
        self.regions_frame = ttk.LabelFrame(main_frame, text="Selected Regions")
        self.regions_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)

        # Create treeview for regions
        self.tree = ttk.Treeview(self.regions_frame, columns=("Region", "Label", "Coordinates", "Time/Freq"), selectmode="extended")
        self.tree.heading("#0", text="ID")
        self.tree.heading("Region", text="Region")
        self.tree.heading("Label", text="Label")
        self.tree.heading("Coordinates", text="Coordinates")
        self.tree.heading("Time/Freq", text="Time/Frequency")

        self.tree.column("#0", width=40)
        self.tree.column("Region", width=80)
        self.tree.column("Label", width=100)
        self.tree.column("Coordinates", width=200)
        self.tree.column("Time/Freq", width=300)

        self.tree.pack(fill=tk.X, expand=True)

        # Delete selected region button
        ttk.Button(self.regions_frame, text="Delete Selected Regions",
                 command=self.delete_selected_region).pack(side=tk.RIGHT, pady=5)

        # Update image counter
        self.update_image_counter()

    def update_image_counter(self):
        """Update the image counter label"""
        if self.image_files:
            self.image_label.config(text=f"Image {self.current_image_idx + 1}/{len(self.image_files)}")
        else:
            self.image_label.config(text="No images found")

    def load_image(self, image_path):
        """Load and display an image in the canvas"""
        try:
            # Load the image
            self.image = Image.open(image_path)

            # Reset zoom
            self.scale_factor = 1.0
            self.zoom_idx = self.zoom_levels.index(1.0)
            self.update_zoom_label()

            # Extract image metadata
            self.image_metadata = self.extract_image_metadata(image_path)

            # Update the photo
            self.photo = ImageTk.PhotoImage(self.image)

            # Update canvas
            self.canvas.delete("all")
            self.canvas.config(scrollregion=(0, 0, self.photo.width(), self.photo.height()))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Draw axes if we have metadata
            if self.image_metadata:
                self.draw_axes()

            # Clear selected regions for the previous image
            self.selected_regions = []

            # Update status bar
            self.status_var.set(f"Loaded image: {os.path.basename(image_path)}")

            # Current image key for the labels data
            self.current_image_key = os.path.basename(image_path)

            # Load regions for this image
            self.load_regions_for_current_image()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def extract_image_metadata(self, image_path):
        """Extract metadata from image filename and bounds file"""
        try:
            # Get the filename without path
            filename = os.path.basename(image_path)

            # Check if we have bounds for this file in the spectrogram_bounds.json
            if filename in self.spectrogram_bounds:
                bounds = self.spectrogram_bounds[filename]

                metadata = {
                    'filename': filename,
                    'time_range': bounds['time_range'],
                    'freq_range': bounds['freq_range'],
                    'spectrogram_region': bounds['spectrogram_region'],
                    'width': self.image.width,
                    'height': self.image.height,
                    'has_bounds': True
                }
                return metadata
            else:
                # If no bounds are available, return limited metadata
                parts = filename.split('_')
                if len(parts) >= 3:
                    patient = ' '.join(parts[:-2])
                    channel = parts[-3]

                    metadata = {
                        'filename': filename,
                        'patient': patient,
                        'channel': channel,
                        'width': self.image.width,
                        'height': self.image.height,
                        'has_bounds': False
                    }
                    return metadata
                return None
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return None

    def draw_axes(self):
        """Draw spectrogram boundaries based on the loaded bounds data"""
        if not self.image_metadata or not self.image_metadata.get('has_bounds', False):
            # If no bounds are available, just return without showing a warning message
            return

        # Draw a rectangle around the spectrogram region if available
        if 'spectrogram_region' in self.image_metadata:
            region = self.image_metadata['spectrogram_region']

            # Scale the coordinates based on current zoom
            x_min = int(region['x_min'] * self.scale_factor)
            y_min = int(region['y_min'] * self.scale_factor)
            x_max = int(region['x_max'] * self.scale_factor)
            y_max = int(region['y_max'] * self.scale_factor)

            # Draw a dashed rectangle around the valid region
            self.canvas.create_rectangle(
                x_min, y_min, x_max, y_max,
                outline="green",
                width=2,
                dash=(5, 5),
                tags="spectrogram_boundary"
            )

            # Add time labels at the bottom
            time_range = self.image_metadata['time_range']
            time_span = time_range[1] - time_range[0]

            # Add time labels
            for i in range(6):
                time_val = time_range[0] + (i * time_span / 5)
                x_pos = x_min + (i * (x_max - x_min) / 5)

                # Add tick marks at the bottom
                self.canvas.create_line(
                    x_pos, y_max, x_pos, y_max + 5,
                    fill="black", tags="time_axis"
                )

                # Add time labels
                self.canvas.create_text(
                    x_pos, y_max + 15,
                    text=f"{time_val:.1f}",
                    fill="black", tags="time_axis"
                )

            # Add "Time (min)" label
            self.canvas.create_text(
                (x_min + x_max) / 2, y_max + 25,
                text="Time (min)",
                fill="black", tags="time_axis"
            )

            # Add frequency labels on the left
            freq_range = self.image_metadata['freq_range']
            freq_span = freq_range[1] - freq_range[0]

            # Add frequency labels
            for i in range(8):
                freq_val = freq_range[0] + (i * freq_span / 7)
                y_pos = y_max - (i * (y_max - y_min) / 7)

                # Add tick marks on the left
                self.canvas.create_line(
                    x_min, y_pos, x_min - 5, y_pos,
                    fill="black", tags="freq_axis"
                )

                # Add frequency labels
                self.canvas.create_text(
                    x_min - 10, y_pos,
                    text=f"{freq_val:.1f}",
                    anchor=tk.E,
                    fill="black", tags="freq_axis"
                )

            # Add "Frequency (Hz)" label
            self.canvas.create_text(
                x_min - 30, (y_min + y_max) / 2,
                text="Frequency (Hz)",
                angle=90,
                fill="black", tags="freq_axis"
            )

            # Add informational status message
            self.status_var.set(
                f"Loaded bounds for {self.image_metadata['filename']} - " +
                f"Time: {time_range[0]}-{time_range[1]} min, Freq: {freq_range[0]}-{freq_range[1]} Hz"
            )

    def get_time_freq_from_coords(self, x, y):
        """Convert pixel coordinates to time and frequency values"""
        if not self.image_metadata or not self.image_metadata.get('has_bounds', False):
            return None

        # Get time and frequency ranges
        time_range = self.image_metadata['time_range']
        freq_range = self.image_metadata['freq_range']

        # Get spectrogram region
        region = self.image_metadata['spectrogram_region']
        x_min = region['x_min']
        y_min = region['y_min']
        x_max = region['x_max']
        y_max = region['y_max']

        # Calculate time from x coordinate
        # Map x from [x_min, x_max] to time_range[0, 1]
        if x < x_min:
            time_val = time_range[0]
        elif x > x_max:
            time_val = time_range[1]
        else:
            time_val = time_range[0] + (x - x_min) * (time_range[1] - time_range[0]) / (x_max - x_min)

        # Calculate frequency from y coordinate
        # Note: y increases downward in canvas, but frequency increases upward
        if y > y_max:
            freq_val = freq_range[0]  # Lowest frequency at bottom
        elif y < y_min:
            freq_val = freq_range[1]  # Highest frequency at top
        else:
            # Map y from [y_min, y_max] to freq_range[1, 0] (note the inversion)
            freq_val = freq_range[1] - (y - y_min) * (freq_range[1] - freq_range[0]) / (y_max - y_min)

        return (time_val, freq_val)

    def on_mouse_wheel(self, event):
        """Handle mouse wheel scrolling for zoom"""
        # Determine zoom direction
        if event.num == 4 or event.delta > 0:
            self.zoom_in(center_x=event.x, center_y=event.y)
        elif event.num == 5 or event.delta < 0:
            self.zoom_out(center_x=event.x, center_y=event.y)

    def zoom_in(self, center_x=None, center_y=None):
        """Zoom in on the image"""
        if self.zoom_idx < len(self.zoom_levels) - 1:
            # Save current view center if not specified
            if center_x is None or center_y is None:
                center_x = self.canvas.winfo_width() / 2
                center_y = self.canvas.winfo_height() / 2

            # Get canvas coordinates before zoom
            x_view = self.canvas.canvasx(center_x)
            y_view = self.canvas.canvasy(center_y)

            # Update zoom level
            self.zoom_idx += 1
            self.scale_factor = self.zoom_levels[self.zoom_idx]

            # Apply zoom
            self.apply_zoom(x_view, y_view)

    def zoom_out(self, center_x=None, center_y=None):
        """Zoom out from the image"""
        if self.zoom_idx > 0:
            # Save current view center if not specified
            if center_x is None or center_y is None:
                center_x = self.canvas.winfo_width() / 2
                center_y = self.canvas.winfo_height() / 2

            # Get canvas coordinates before zoom
            x_view = self.canvas.canvasx(center_x)
            y_view = self.canvas.canvasy(center_y)

            # Update zoom level
            self.zoom_idx -= 1
            self.scale_factor = self.zoom_levels[self.zoom_idx]

            # Apply zoom
            self.apply_zoom(x_view, y_view)

    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.zoom_idx = self.zoom_levels.index(1.0)
        self.scale_factor = 1.0
        self.apply_zoom()

    def apply_zoom(self, center_x=None, center_y=None):
        """Apply the current zoom level to the image"""
        if self.image is None:
            return

        # Create new zoomed image
        new_width = int(self.image.width * self.scale_factor)
        new_height = int(self.image.height * self.scale_factor)

        # Resample with better quality for larger sizes, BILINEAR for smaller
        resample = Image.LANCZOS if self.scale_factor > 1.0 else Image.BILINEAR

        # Resize image
        zoomed_image = self.image.resize((new_width, new_height), resample)
        self.photo = ImageTk.PhotoImage(zoomed_image)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Redraw axes
        if self.image_metadata:
            self.draw_axes()

        # Redraw all regions with scaled coordinates
        self.redraw_regions_with_zoom()

        # Center view on the specified point
        if center_x is not None and center_y is not None:
            # Calculate new position after zoom
            new_x = center_x * self.scale_factor / self.zoom_levels[self.zoom_idx - 1] if self.zoom_idx > 0 else center_x
            new_y = center_y * self.scale_factor / self.zoom_levels[self.zoom_idx - 1] if self.zoom_idx > 0 else center_y

            # Calculate fractions for xview and yview
            x_fraction = new_x / new_width
            y_fraction = new_y / new_height

            # Center the view on this point
            self.canvas.xview_moveto(max(0, x_fraction - 0.5))
            self.canvas.yview_moveto(max(0, y_fraction - 0.5))

        # Update zoom label
        self.update_zoom_label()

    def fit_to_window(self):
        """Resize the image to fit the canvas window"""
        if self.image is None:
            return

        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # If canvas is not yet properly sized, we can't do much
        if canvas_width <= 1 or canvas_height <= 1:
            return

        # Calculate the scale factor needed to fit the image in the canvas
        width_scale = canvas_width / self.image.width
        height_scale = canvas_height / self.image.height

        # Use the smaller scale to ensure the entire image fits
        new_scale = min(width_scale, height_scale) * 0.95  # 95% of actual fit for a small margin

        # Find the closest zoom level
        closest_idx = min(range(len(self.zoom_levels)),
                          key=lambda i: abs(self.zoom_levels[i] - new_scale))

        # Set the zoom level and apply
        self.zoom_idx = closest_idx
        self.scale_factor = self.zoom_levels[closest_idx]

        # Apply the zoom without a specific center point
        self.apply_zoom()

        # Center the image in the canvas
        self.canvas.xview_moveto(0.5 - (canvas_width / (self.image.width * self.scale_factor * 2)))
        self.canvas.yview_moveto(0.5 - (canvas_height / (self.image.height * self.scale_factor * 2)))

    def update_zoom_label(self):
        """Update zoom percentage label"""
        zoom_percent = int(self.scale_factor * 100)
        self.zoom_label.config(text=f"{zoom_percent}%")

    def redraw_regions_with_zoom(self):
        """Redraw all regions with proper scaling for current zoom level"""
        for i, region in enumerate(self.selected_regions):
            # Scale the coordinates
            x1 = int(region['x1'] * self.scale_factor)
            y1 = int(region['y1'] * self.scale_factor)
            x2 = int(region['x2'] * self.scale_factor)
            y2 = int(region['y2'] * self.scale_factor)

            # Draw with proper color
            color = "blue" if region['label'] == "hypo precursor" else "red"

            # Create rectangle
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color, width=2, tags=f"region_{i+1}"
            )

            # Add label text
            self.canvas.create_text(
                x1 + 5, y1 + 5,
                text=f"R{i+1}: {region['label']}",
                anchor=tk.NW,
                fill=color,
                tags=f"region_{i+1}"
            )

    def start_pan(self, event):
        """Start panning with middle mouse button"""
        self.canvas.scan_mark(event.x, event.y)

    def pan(self, event):
        """Pan the canvas with middle mouse button drag"""
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def center_view(self, event):
        """Center the view on the double-clicked point"""
        # Get canvas coordinates
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Calculate fractions for xview and yview
        x_fraction = x / self.canvas.bbox("all")[2]
        y_fraction = y / self.canvas.bbox("all")[3]

        # Center the view on this point
        self.canvas.xview_moveto(max(0, x_fraction - 0.5))
        self.canvas.yview_moveto(max(0, y_fraction - 0.5))

        # If we have metadata, show time/frequency at this point
        if self.image_metadata:
            time_freq = self.get_time_freq_from_coords(x / self.scale_factor, y / self.scale_factor)
            if time_freq:
                self.status_var.set(f"Time: {time_freq[0]:.2f} min, Frequency: {time_freq[1]:.2f} Hz")

    def next_image(self):
        """Load the next image"""
        if not self.image_files:
            return

        if self.current_image_idx < len(self.image_files) - 1:
            self.current_image_idx += 1
            self.load_image(self.image_files[self.current_image_idx])
            self.update_image_counter()

    def prev_image(self):
        """Load the previous image"""
        if not self.image_files:
            return

        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.load_image(self.image_files[self.current_image_idx])
            self.update_image_counter()

    def clear_current_image(self):
        """Clear all regions for the current image"""
        if messagebox.askyesno("Confirm", "Clear all regions for this image?"):
            if self.current_image_key in self.labels_data:
                del self.labels_data[self.current_image_key]
            self.selected_regions = []
            self.load_regions_for_current_image()
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            # Redraw axes if we have metadata
            if self.image_metadata:
                self.draw_axes()
            self.status_var.set(f"Cleared all regions for {self.current_image_key}")

    def export_to_csv(self):
        """Export labels to CSV format"""
        if not self.labels_data:
            messagebox.showinfo("Info", "No labels to export")
            return

        try:
            rows = []
            for image_name, regions in self.labels_data.items():
                for region in regions:
                    # Extract patient and channel from filename
                    parts = image_name.split('_')
                    if len(parts) >= 3:
                        patient = ' '.join(parts[:-2])  # Assuming format is "Insulin Clamp #X_Channel_spectrogram_with_signal.png"
                        channel = parts[-3]
                    else:
                        patient = "Unknown"
                        channel = "Unknown"

                    # Add time/frequency information if available
                    time_freq_info = ""
                    if self.image_metadata:
                        time1, freq1 = self.get_time_freq_from_coords(region['x1'], region['y1'])
                        time2, freq2 = self.get_time_freq_from_coords(region['x2'], region['y2'])
                        time_freq_info = f"Time: {min(time1, time2):.2f}-{max(time1, time2):.2f} min, Freq: {min(freq1, freq2):.2f}-{max(freq1, freq2):.2f} Hz"

                    rows.append({
                        'image_name': image_name,
                        'patient': patient,
                        'channel': channel,
                        'label': region['label'],
                        'x1': region['x1'],
                        'y1': region['y1'],
                        'x2': region['x2'],
                        'y2': region['y2'],
                        'time_min': min(time1, time2) if self.image_metadata else None,
                        'time_max': max(time1, time2) if self.image_metadata else None,
                        'freq_min': min(freq1, freq2) if self.image_metadata else None,
                        'freq_max': max(freq1, freq2) if self.image_metadata else None,
                        'time_freq_info': time_freq_info,
                        'timestamp': region.get('timestamp', datetime.now().isoformat())
                    })

            # Create DataFrame
            df = pd.DataFrame(rows)

            # Ask user for save location
            csv_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialfile="spectrogram_labels.csv",
                title="Save CSV file",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            if not csv_path:  # User cancelled
                return

            # Save to CSV
            df.to_csv(csv_path, index=False)
            messagebox.showinfo("Success", f"Labels exported to {csv_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV: {str(e)}")
            print(f"Error details: {e}")

    def on_mouse_down(self, event):
        """Handle mouse button press"""
        # If no bounds are available, disable labeling
        if not self.image_metadata or not self.image_metadata.get('has_bounds', False):
            self.status_var.set("Labeling is disabled for this image - no spectrogram bounds available")
            return

        # Convert canvas coordinates to image coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Convert to unscaled image coordinates
        img_x = canvas_x / self.scale_factor
        img_y = canvas_y / self.scale_factor

        # Check if the point is within the valid spectrogram region
        region = self.image_metadata['spectrogram_region']
        if not (region['x_min'] <= img_x <= region['x_max'] and
                region['y_min'] <= img_y <= region['y_max']):
            self.status_var.set("Cannot start labeling outside the valid spectrogram region")
            return

        # Scale the region bounds to match the current zoom level
        x_min = region['x_min'] * self.scale_factor
        y_min = region['y_min'] * self.scale_factor
        x_max = region['x_max'] * self.scale_factor
        y_max = region['y_max'] * self.scale_factor

        # Constrain the starting point to the valid region
        canvas_x = max(x_min, min(canvas_x, x_max))
        canvas_y = max(y_min, min(canvas_y, y_max))

        self.start_x = canvas_x
        self.start_y = canvas_y
        self.drawing = True

        # Create a new rectangle
        self.rect = self.canvas.create_rectangle(
            canvas_x, canvas_y, canvas_x, canvas_y,
            outline="green", width=2, dash=(4, 4)
        )

    def on_mouse_move(self, event):
        """Handle mouse movement while button is pressed"""
        if not self.drawing or not self.rect:
            return

        # Convert canvas coordinates to image coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # If we have bounds, constrain the coordinates to the valid region
        if self.image_metadata and self.image_metadata.get('has_bounds', False):
            region = self.image_metadata['spectrogram_region']

            # Scale the region bounds to match the current zoom level
            x_min = region['x_min'] * self.scale_factor
            y_min = region['y_min'] * self.scale_factor
            x_max = region['x_max'] * self.scale_factor
            y_max = region['y_max'] * self.scale_factor

            # Constrain the coordinates to the valid region
            canvas_x = max(x_min, min(canvas_x, x_max))
            canvas_y = max(y_min, min(canvas_y, y_max))

        # Update rectangle
        self.canvas.coords(self.rect, self.start_x, self.start_y, canvas_x, canvas_y)

        # Update coordinate display
        self.update_coordinate_display(canvas_x, canvas_y)

    def on_mouse_up(self, event):
        """Handle mouse button release"""
        if not self.drawing:
            return

        self.drawing = False

        # Convert canvas coordinates to image coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Ensure proper coordinates (x1 < x2, y1 < y2)
        x1 = min(self.start_x, canvas_x)
        y1 = min(self.start_y, canvas_y)
        x2 = max(self.start_x, canvas_x)
        y2 = max(self.start_y, canvas_y)

        # Check if the rectangle is too small
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            self.canvas.delete(self.rect)
            return

        # Delete the temporary rectangle
        self.canvas.delete(self.rect)

        # Create the final rectangle with proper color based on label
        color = "blue" if self.current_label.get() == "hypo precursor" else "red"

        # Get time/frequency information if available
        time_freq_info = ""
        if self.image_metadata:
            # Convert pixel coordinates to time and frequency values
            time1, freq1 = self.get_time_freq_from_coords(x1 / self.scale_factor, y1 / self.scale_factor)
            time2, freq2 = self.get_time_freq_from_coords(x2 / self.scale_factor, y2 / self.scale_factor)
            time_freq_info = f"Time: {min(time1, time2):.2f}-{max(time1, time2):.2f} min, Freq: {min(freq1, freq2):.2f}-{max(freq1, freq2):.2f} Hz"

        # Store the region
        region = {
            'x1': int(x1 / self.scale_factor),
            'y1': int(y1 / self.scale_factor),
            'x2': int(x2 / self.scale_factor),
            'y2': int(y2 / self.scale_factor),
            'label': self.current_label.get(),
            'time_freq_info': time_freq_info,
            'timestamp': datetime.now().isoformat()
        }

        self.selected_regions.append(region)

        # Save to the labels data
        if self.current_image_key not in self.labels_data:
            self.labels_data[self.current_image_key] = []

        self.labels_data[self.current_image_key] = self.selected_regions

        # Update the treeview
        region_id = len(self.selected_regions)
        self.tree.insert("", tk.END, text=str(region_id),
                       values=(f"Region {region_id}", region['label'],
                              f"({region['x1']}, {region['y1']}) to ({region['x2']}, {region['y2']})",
                              time_freq_info))

        # Draw the region on canvas with scaled coordinates
        x1 = int(region['x1'] * self.scale_factor)
        y1 = int(region['y1'] * self.scale_factor)
        x2 = int(region['x2'] * self.scale_factor)
        y2 = int(region['y2'] * self.scale_factor)

        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline=color, width=2, tags=f"region_{region_id}"
        )

        # Add label text
        self.canvas.create_text(
            x1 + 5, y1 + 5,
            text=f"R{region_id}: {region['label']}",
            anchor=tk.NW,
            fill=color,
            tags=f"region_{region_id}"
        )

        # Update status
        self.status_var.set(f"Added {region['label']} region at ({x1}, {y1}) to ({x2}, {y2})" +
                          (f" - {time_freq_info}" if time_freq_info else ""))

    def delete_selected_region(self):
        """Delete the selected regions from treeview and canvas"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Info", "No regions selected")
            return

        if len(selection) > 1:
            # Confirm deletion of multiple regions
            if not messagebox.askyesno("Confirm", f"Delete {len(selection)} selected regions?"):
                return

        # Get the region IDs from the selection
        region_ids = [int(self.tree.item(item, "text")) for item in selection]

        # Sort in descending order to avoid index shifting during removal
        region_ids.sort(reverse=True)

        # Remove the regions
        for region_id in region_ids:
            # Convert from 1-based to 0-based index
            idx = region_id - 1
            if 0 <= idx < len(self.selected_regions):
                # Remove from data
                self.selected_regions.pop(idx)

        # Update labels data
        if self.current_image_key in self.labels_data:
            self.labels_data[self.current_image_key] = self.selected_regions

        # Refresh the display
        self.load_regions_for_current_image()

        # Redraw the image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Redraw axes if we have metadata
        if self.image_metadata:
            self.draw_axes()

        # Redraw all regions
        self.redraw_regions_with_zoom()

        # Update status
        if len(region_ids) == 1:
            self.status_var.set(f"Deleted region {region_ids[0]}")
        else:
            self.status_var.set(f"Deleted {len(region_ids)} regions")

    def load_regions_for_current_image(self):
        """Load and display regions for the current image"""
        # Clear treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Load regions from saved data
        if self.current_image_key in self.labels_data:
            self.selected_regions = self.labels_data[self.current_image_key]

            # Add regions to treeview and canvas
            for i, region in enumerate(self.selected_regions):
                # Get time/freq info if available
                time_freq_info = region.get('time_freq_info', '')

                # Add to treeview
                self.tree.insert("", tk.END, text=str(i+1),
                               values=(f"Region {i+1}", region['label'],
                                      f"({region['x1']}, {region['y1']}) to ({region['x2']}, {region['y2']})",
                                      time_freq_info))

                # Draw on canvas
                self.draw_saved_region(region, i+1)

    def draw_saved_region(self, region, region_id):
        """Draw a saved region on the canvas"""
        # Scale the coordinates
        x1 = int(region['x1'] * self.scale_factor)
        y1 = int(region['y1'] * self.scale_factor)
        x2 = int(region['x2'] * self.scale_factor)
        y2 = int(region['y2'] * self.scale_factor)

        # Draw with proper color
        color = "blue" if region['label'] == "hypo precursor" else "red"

        # Create rectangle
        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline=color, width=2, tags=f"region_{region_id}"
        )

        # Add label text
        self.canvas.create_text(
            x1 + 5, y1 + 5,
            text=f"R{region_id}: {region['label']}",
            anchor=tk.NW,
            fill=color,
            tags=f"region_{region_id}"
        )

    def load_spectrogram_bounds(self):
        """Load spectrogram bounds from JSON file"""
        if self.bounds_file.exists():
            try:
                with open(self.bounds_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                messagebox.showwarning("Warning", "Bounds file is corrupted. Using defaults.")
                return {}
        else:
            print(f"Bounds file not found: {self.bounds_file}")
            return {}

    def open_file_dialog(self):
        """Open a file dialog to select a specific image file"""
        file_path = filedialog.askopenfilename(
            title="Open Spectrogram File",
            initialdir=self.spectrograms_dir,
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )

        if file_path:
            # Check if the file is already in our list
            if file_path in self.image_files:
                self.current_image_idx = self.image_files.index(file_path)
            else:
                # Add the new file to the list and set as current
                self.image_files.append(file_path)
                self.current_image_idx = len(self.image_files) - 1

            # Load the selected image
            self.load_image(file_path)
            self.update_image_counter()

            # Auto-fit the image to window
            self.root.update()
            self.fit_to_window()

            # Update status
            self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")

    def on_window_resize(self, event):
        """Handle window resize event"""
        # Get new size
        new_width = event.width
        new_height = event.height

        # Update canvas size
        self.canvas_frame.config(width=new_width, height=new_height)

        # Refit image to window
        self.fit_to_window()

    def update_coordinate_display(self, x, y):
        """Update the display of cursor coordinates"""
        if self.coord_text_id:
            self.canvas.delete(self.coord_text_id)

        # Convert to image coordinates
        img_x = x / self.scale_factor
        img_y = y / self.scale_factor

        # Get time and frequency from coordinates
        time_freq = self.get_time_freq_from_coords(img_x, img_y)
        if time_freq:
            time_val, freq_val = time_freq
            coord_text = f"Time: {time_val:.2f} min, Freq: {freq_val:.2f} Hz"
        else:
            coord_text = f"X: {img_x:.1f}, Y: {img_y:.1f}"

        # Display coordinates at the cursor position
        self.coord_text_id = self.canvas.create_text(
            x + 10, y + 10,
            text=coord_text,
            anchor=tk.NW,
            fill="black",
            font=("Arial", 10, "normal"),
            tags="coord_display"
        )

    def on_cursor_move(self, event):
        """Handle mouse motion event for coordinate display"""
        # Get canvas coordinates
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        # Update coordinate display
        self.update_coordinate_display(x, y)

def main():
    root = tk.Tk()
    app = SpectrogramLabeler(root)
    root.mainloop()

if __name__ == "__main__":
    main()
