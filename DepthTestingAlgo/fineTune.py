import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
import os
from PIL import Image, ImageTk

class StereoDepthTuner:
    def __init__(self, root):
        self.root = root
        self.root.title("Stereo Depth Tuner")  #set the main window
        
        # Default paths (modify these as needed)
        self.DEFAULT_PATHS = {
            'left_image': "/home/jetson/camera/Depth_estimation/calibrationImages/test/imageLeft0.png",
            'right_image': "/home/jetson/camera/Depth_estimation/calibrationImages/test/imageRight0.png",
            'calib_dir': "/home/jetson/camera/Depth_estimation"
        }
        
        # For Python 3.6 compatibility, use geometry instead of zoomed
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{int(screen_width*0.9)}x{int(screen_height*0.8)}")
        
        # Initialize variables
        self.left_image = None
        self.right_image = None
        self.calib_dir = None
        self.left_map = None
        self.right_map = None
        self.Q = None
        self.processed = False
        
        self.left_map_x = None
        self.left_map_y = None
        self.right_map_x = None
        self.right_map_y = None
        self.left_rectified = None
        self.right_rectified = None

        # Rectified image holders
        self.left_rectified = None
        self.right_rectified = None
        
        # SGBM default parameters
        self.min_disparity = 0
        self.num_disparities = 128
        self.block_size = 11
        self.p1 = 8 * 3 * 3
        self.p2 = 32 * 3 * 3
        self.disp12_max_diff = 1
        self.pre_filter_cap = 63
        self.uniqueness_ratio = 15
        self.speckle_window_size = 200
        self.speckle_range = 32
        
        # Filter parameters
        self.wls_lambda = 8000.0
        self.wls_sigma = 0.8
        self.roi_scale = 0.5
        
        # Create mode variable
        self.mode_var = tk.IntVar(value=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        
        # Initialize GUI components
        self.initialize_gui()
        
        # Try to load default images if they exist
        self.try_load_defaults()
    
    def try_load_defaults(self):
        """Attempt to load default paths if they exist"""
        try:
            if os.path.exists(self.DEFAULT_PATHS['left_image']):
                self.left_image = cv2.imread(self.DEFAULT_PATHS['left_image'])
                self.status_label.config(text="Loaded default left image", fg="green")
            
            if os.path.exists(self.DEFAULT_PATHS['right_image']):
                self.right_image = cv2.imread(self.DEFAULT_PATHS['right_image'])
                self.status_label.config(text="Loaded default right image", fg="green")
            
            if os.path.exists(self.DEFAULT_PATHS['calib_dir']):
                self.calib_dir = self.DEFAULT_PATHS['calib_dir']
                self.load_calibration_data()
                self.status_label.config(text="Loaded default calibration", fg="green")
            
            if self.left_image is not None and self.right_image is not None:
                self.rectify_images()
                self.show_images()
        except Exception as e:
            self.status_label.config(text=f"Error loading defaults: {str(e)}", fg="red")
    
    def initialize_gui(self):
        """Initialize all GUI components with improved layout"""
        # Main container
        main_container = tk.PanedWindow(self.root, orient=tk.VERTICAL)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Top panel for images (2x2 grid)
        image_panel = tk.Frame(main_container)
        main_container.add(image_panel, height=600)
        
        # Create 2x2 grid for images with scrollable canvases
        self.create_scrollable_image(image_panel, "left", 0, 0)
        self.create_scrollable_image(image_panel, "right", 0, 1)
        self.create_scrollable_image(image_panel, "depth", 1, 0)
        self.create_scrollable_image(image_panel, "disparity", 1, 1)
        
        # Configure grid weights
        image_panel.grid_rowconfigure(0, weight=1)
        image_panel.grid_rowconfigure(1, weight=1)
        image_panel.grid_columnconfigure(0, weight=1)
        image_panel.grid_columnconfigure(1, weight=1)
        
        # Add titles to each image box
        self.add_image_titles(image_panel)
        
        # Bottom panel for controls
        control_panel = tk.Frame(main_container)
        main_container.add(control_panel)
        
        # Control frame
        control_frame = tk.LabelFrame(control_panel, text="Controls", padx=5, pady=5)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Image selection buttons
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(btn_frame, text="Select Left Image", command=self.load_left_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Select Right Image", command=self.load_right_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Select Calibration", command=self.load_calibration).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Process Images", command=self.process_images).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save Images", command=self.saveImages).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Load calibration data and images to begin", fg="blue")
        self.status_label.pack(fill=tk.X, padx=5, pady=2)
        
        # Parameter controls notebook (tabs)
        param_notebook = ttk.Notebook(control_frame)
        param_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs for different parameter groups
        sgbm_frame = ttk.Frame(param_notebook)
        filter_frame = ttk.Frame(param_notebook)
        param_notebook.add(sgbm_frame, text="SGBM Parameters")
        param_notebook.add(filter_frame, text="Filter Parameters")
        
        # SGBM parameters
        self.create_slider(sgbm_frame, "min_disparity", -50, 50, 1, row=0)
        self.create_slider(sgbm_frame, "num_disparities", 16, 256, 16, row=1)
        self.create_slider(sgbm_frame, "block_size", 1, 21, 2, row=2)
        self.create_slider(sgbm_frame, "p1", 0, 1000, 1, row=3)
        self.create_slider(sgbm_frame, "p2", 0, 5000, 1, row=4)
        self.create_slider(sgbm_frame, "disp12_max_diff", 0, 100, 1, row=5)
        self.create_slider(sgbm_frame, "pre_filter_cap", 1, 100, 1, row=6)
        
        # Filter parameters
        self.create_slider(filter_frame, "uniqueness_ratio", 0, 100, 1, row=0)
        self.create_slider(filter_frame, "speckle_window_size", 0, 500, 1, row=1)
        self.create_slider(filter_frame, "speckle_range", 0, 100, 1, row=2)
        self.create_slider(filter_frame, "wls_lambda", 0, 20000, 100, row=3)
        self.create_slider(filter_frame, "wls_sigma", 0.1, 5.0, 0.1, row=4)
        self.create_slider(filter_frame, "roi_scale", 0.1, 1.0, 0.1, row=5)
        
        # Mode selection
        mode_frame = tk.LabelFrame(sgbm_frame, text="SGBM Mode", padx=5, pady=5)
        mode_frame.grid(row=7, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        tk.Radiobutton(mode_frame, text="SGBM", variable=self.mode_var, 
                      value=cv2.STEREO_SGBM_MODE_SGBM).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(mode_frame, text="SGBM 3WAY", variable=self.mode_var, 
                      value=cv2.STEREO_SGBM_MODE_SGBM_3WAY).pack(side=tk.LEFT, padx=10)
    
    def create_scrollable_image(self, parent, img_type, row, col):
        """Create a scrollable image canvas with zoom capability"""
        # Create frame to hold canvas and scrollbars
        frame = tk.Frame(parent, borderwidth=2, relief="groove")
        frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
        
        # Create canvas with scrollbars
        canvas = tk.Canvas(frame)
        h_scroll = tk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
        v_scroll = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        # Layout
        h_scroll.pack(side="bottom", fill="x")
        v_scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Create image placeholder on canvas
        img_id = canvas.create_image(0, 0, anchor="nw")
        
        # Bind click event for fullscreen view
        canvas.bind("<Button-1>", lambda e, t=img_type: self.show_fullscreen(t))
        
        # Store references
        setattr(self, f"{img_type}_canvas", canvas)
        setattr(self, f"{img_type}_img_id", img_id)
        setattr(self, f"{img_type}_frame", frame)
    
    def show_fullscreen(self, img_type):
        """Show selected image in fullscreen mode"""
        # Get the image to display
        if img_type == "left" and self.left_rectified is not None:
            img = self.left_rectified.copy()
            title = "Left Image (Rectified)"
        elif img_type == "right" and self.right_rectified is not None:
            img = self.right_rectified.copy()
            title = "Right Image (Rectified)"
        elif img_type == "depth" and hasattr(self, 'depth_img'):
            img = cv2.cvtColor(np.array(self.depth_img), cv2.COLOR_RGB2BGR)
            title = "Depth Map"
        elif img_type == "disparity" and hasattr(self, 'disparity_img'):
            img = cv2.cvtColor(np.array(self.disparity_img), cv2.COLOR_RGB2BGR)
            title = "Disparity Map"
        else:
            return
        
        # Create new window
        top = tk.Toplevel(self.root)
        top.title(title)
        
        # Get screen dimensions
        screen_width = top.winfo_screenwidth()
        screen_height = top.winfo_screenheight()
        
        # Create scrollable canvas
        canvas = tk.Canvas(top)
        h_scroll = tk.Scrollbar(top, orient="horizontal", command=canvas.xview)
        v_scroll = tk.Scrollbar(top, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        # Layout
        h_scroll.pack(side="bottom", fill="x")
        v_scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Convert and display image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Create image on canvas
        canvas.create_image(0, 0, anchor="nw", image=img_tk)
        canvas.config(scrollregion=canvas.bbox("all"))
        
        # Keep reference to avoid garbage collection
        canvas.image = img_tk
    
    def add_image_titles(self, parent):
        """Add descriptive titles to each image box"""
        titles = ["Left Image (Rectified)", "Right Image (Rectified)", "Depth Map", "Disparity Map"]
        positions = [(0,0), (0,1), (1,0), (1,1)]
        
        for (row, col), title in zip(positions, titles):
            lbl = tk.Label(parent, text=title, bg="white", fg="black")
            lbl.grid(row=row, column=col, sticky="nw", padx=10, pady=5)
    
    def create_slider(self, parent, param_name, from_, to, resolution, row):
        """Create a slider control for a parameter with improved layout"""
        frame = tk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", padx=5, pady=2)
        
        current_value = getattr(self, param_name)
        
        # Parameter name label
        tk.Label(frame, text=param_name.replace("_", " ").title() + ":", 
                width=20, anchor="e").pack(side=tk.LEFT)
        
        # Slider
        scale = tk.Scale(
            frame, 
            from_=from_, 
            to=to, 
            resolution=resolution,
            orient=tk.HORIZONTAL,
            command=lambda v, p=param_name: self.on_slider_change(p, float(v)),
            length=200
        )
        scale.set(current_value)
        scale.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # Current value display
        value_label = tk.Label(frame, text=str(current_value), width=10)
        value_label.pack(side=tk.LEFT, padx=5)
        
        # Store references
        setattr(self, f"{param_name}_scale", scale)
        setattr(self, f"{param_name}_label", value_label)
    
    def on_slider_change(self, param_name, value):
        """Handle slider value changes"""
        setattr(self, param_name, value)
        # Update value label
        getattr(self, f"{param_name}_label").config(text=f"{value:.1f}")
        if self.processed:
            self.process_images()
    
    def load_left_image(self):
        """Load left stereo image"""
        file_path = filedialog.askopenfilename(
            title="Select Left Image",
            initialdir=os.path.dirname(self.DEFAULT_PATHS['left_image'])
        )
        
        if file_path:
            self.left_image = cv2.imread(file_path)
            self.rectify_images()
            self.show_images()
            self.update_status()
    
    def load_right_image(self):
        """Load right stereo image"""
        file_path = filedialog.askopenfilename(
            title="Select Right Image",
            initialdir=os.path.dirname(self.DEFAULT_PATHS['right_image'])
        )
        if file_path:
            self.right_image = cv2.imread(file_path)
            self.rectify_images()
            self.show_images()
            self.update_status()
    
#     def load_calibration(self):
#         """Load calibration data"""
#         calib_dir = filedialog.askdirectory(
#             title="Select Calibration Directory",
#             initialdir=self.DEFAULT_PATHS['calib_dir']
#         )
#         if calib_dir:
#             self.calib_dir = calib_dir
#             self.load_calibration_data()
    def load_calibration(self):
        calib_path = filedialog.askopenfilename(title="Select calibration_data.npz", filetypes=[("NumPy files", "*.npz")])
        rect_path = filedialog.askopenfilename(title="Select rectification_maps.npz", filetypes=[("NumPy files", "*.npz")])

        if calib_path and rect_path:
            try:
                calibration_data = np.load(calib_path)
                rectification_maps = np.load(rect_path)

                # Store maps for later use
                self.left_map_x = rectification_maps['left_map_x']
                self.left_map_y = rectification_maps['left_map_y']
                self.right_map_x = rectification_maps['right_map_x']
                self.right_map_y = rectification_maps['right_map_y']

                messagebox.showinfo("Success", "Calibration and rectification maps loaded.")

                # Optional: rectify existing loaded images immediately
                if self.left_image is not None and self.right_image is not None:
                    self.left_rectified = cv2.remap(self.left_image, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR)
                    self.right_rectified = cv2.remap(self.right_image, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)

                    self.display_image(self.left_rectified, self.left_canvas)
                    self.display_image(self.right_rectified, self.right_canvas)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load maps: {e}")

    def load_calibration_data(self):
        """Load calibration data from .npz file"""
        if self.calib_dir:
            try:
                # Look for .npz files in the directory
                npz_files = [f for f in os.listdir(self.calib_dir) if f.endswith('.npz')]
                if not npz_files:
                    raise FileNotFoundError("No .npz calibration file found")

                # Load the first .npz file found
                calib_file = os.path.join(self.calib_dir, npz_files[0])
                calibration_data = np.load(calib_file)

                # Load maps (using 16SC2 format for CPU optimization)
                self.left_map = (
                    calibration_data['left_map_16SC2_x'],
                    calibration_data['left_map_16SC2_y']
                )
                self.right_map = (
                    calibration_data['right_map_16SC2_x'],
                    calibration_data['right_map_16SC2_y']
                )

                # Load Q matrix
                self.Q = calibration_data['Q']

                print("Calibration data loaded successfully from", calib_file)
                self.status_label.config(text="Calibration loaded successfully", fg="green")

                # Rectify images if they're already loaded
                if self.left_image is not None and self.right_image is not None:
                    self.rectify_images()
                    self.show_images()

            except Exception as e:
                print(f"Error loading calibration data: {e}")
                self.status_label.config(text=f"Error loading calibration: {str(e)}", fg="red")

    def rectify_images(self):
        """Rectify images if calibration data is available"""
        if self.left_image is not None and self.right_image is not None:
            if self.left_map is not None and self.right_map is not None:
                self.left_rectified = cv2.remap(self.left_image, self.left_map[0], self.left_map[1], cv2.INTER_LINEAR)
                self.right_rectified = cv2.remap(self.right_image, self.right_map[0], self.right_map[1], cv2.INTER_LINEAR)
            else:
                # If no calibration data, use original images
                self.left_rectified = self.left_image.copy()
                self.right_rectified = self.right_image.copy()
    
    def update_status(self):
        """Update the status label based on current state"""
        if self.left_image is None or self.right_image is None:
            self.status_label.config(text="Please load both left and right images", fg="red")
        elif self.left_map is None or self.right_map is None:
            self.status_label.config(text="Using raw images (no calibration data loaded)", fg="orange")
        else:
            self.status_label.config(text="Using rectified images (calibration data loaded)", fg="green")
            
            
    def saveImages(self):
       
        if self.left_rectified is not None:
            cv2.imwrite(f"left_rectified.png", self.left_rectified)
            print("Left image saved.")

        if self.right_rectified is not None:
            cv2.imwrite(f"right_rectified.png", self.right_rectified)
            print("Right image saved.")
            
            
            
    def show_images(self):
        """Display the loaded images in the 2x2 grid"""
        if self.left_rectified is not None:
            self.display_image_on_canvas("left", self.left_rectified)
        
        if self.right_rectified is not None:
            self.display_image_on_canvas("right", self.right_rectified)
    
    def display_image_on_canvas(self, img_type, cv_img):
        """Display an OpenCV image on the specified canvas"""
        canvas = getattr(self, f"{img_type}_canvas")
        img_id = getattr(self, f"{img_type}_img_id")
        
        # Convert to RGB and resize for display
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Get canvas dimensions
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Resize image to fit canvas while maintaining aspect ratio
        img_ratio = img_pil.width / img_pil.height
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            # Image is wider than canvas
            new_width = canvas_width
            new_height = int(new_width / img_ratio)
        else:
            # Image is taller than canvas
            new_height = canvas_height
            new_width = int(new_height * img_ratio)
        
        # Resize if needed
        if new_width > 0 and new_height > 0:
            img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Update canvas
        canvas.itemconfig(img_id, image=img_tk)
        canvas.config(scrollregion=canvas.bbox("all"))
        
        # Keep reference to avoid garbage collection
        setattr(self, f"{img_type}_img", img_tk)
    
    def process_images(self):
        """Process stereo images to generate depth map"""
        if self.left_rectified is None or self.right_rectified is None:
            print("Please load both left and right images first")
            self.status_label.config(text="Please load both images first", fg="red")
            return
            
        print("Processing images with current parameters...")
        self.status_label.config(text="Processing images...", fg="blue")
        self.root.update()  # Force UI update
        
        # Convert to grayscale
        gray_left = cv2.cvtColor(self.left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(self.right_rectified, cv2.COLOR_BGR2GRAY)
        
        # Create SGBM matcher
        window_size = max(1, int(self.block_size))
        if window_size % 2 == 0:
            window_size += 1
            
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=int(self.min_disparity),
            numDisparities=int(self.num_disparities),
            blockSize=window_size,
            P1=int(self.p1),
            P2=int(self.p2),
            disp12MaxDiff=int(self.disp12_max_diff),
            preFilterCap=int(self.pre_filter_cap),
            uniquenessRatio=int(self.uniqueness_ratio),
            speckleWindowSize=int(self.speckle_window_size),
            speckleRange=int(self.speckle_range),
            mode=int(self.mode_var.get())
        )
        
        # Compute disparity
        left_disp = left_matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        # Create right matcher for WLS filter
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        right_disp = right_matcher.compute(gray_right, gray_left).astype(np.float32) / 16.0
        
        # WLS filter
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        wls_filter.setLambda(self.wls_lambda)
        wls_filter.setSigmaColor(self.wls_sigma)
        filtered_disp = wls_filter.filter(left_disp, gray_left, disparity_map_right=right_disp)
        
        # Apply ROI if needed
        if self.roi_scale < 1.0:
            h, w = filtered_disp.shape
            roi = (
                int(w*(1-self.roi_scale)//2), 
                int(h*(1-self.roi_scale)//2), 
                int(w*self.roi_scale), 
                int(h*self.roi_scale)
            )
            filtered_disp = filtered_disp[
                roi[1]:roi[1]+roi[3], 
                roi[0]:roi[0]+roi[2]
            ]
        
        # Normalize for display
        disp_norm = cv2.normalize(filtered_disp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
        
        # Show raw disparity (before WLS filter)
        raw_disp_norm = cv2.normalize(left_disp, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        raw_disp_color = cv2.applyColorMap(raw_disp_norm, cv2.COLORMAP_JET)
        
        # Show all images
        self.show_images()
        
        # Show depth map
        img = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)
        self.depth_img = Image.fromarray(img)
        self.display_image_on_canvas("depth", disp_color)
        
        # Show disparity map
        disp_img = cv2.cvtColor(raw_disp_color, cv2.COLOR_BGR2RGB)
        self.disparity_img = Image.fromarray(disp_img)
        self.display_image_on_canvas("disparity", raw_disp_color)
        
        self.processed = True
        self.status_label.config(text="Processing complete", fg="green")

if __name__ == "__main__":
    root = tk.Tk()
    app = StereoDepthTuner(root)
    root.mainloop()