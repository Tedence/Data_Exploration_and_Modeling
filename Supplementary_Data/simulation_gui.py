"""
GUI Application for Dynamic T1D ATP-Synthase Magnetic Field Simulation

This application provides a graphical interface for controlling
all simulation parameters and visualizing results in real-time.

Features:
- Interactive parameter controls for all physiological coefficients
- Real-time simulation execution with progress tracking
- Multi-panel visualization of glucose, insulin, ATP, and magnetic field data
- Export capabilities for results and plots
- Parameter preset management
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import threading
import json
import numpy as np
from datetime import datetime
from scipy.signal import welch, spectrogram
from scipy.stats import linregress

# Import the simulation components
from simulation_dynamic import (
    DynamicT1DSimulator, create_atp_sources,
    simulate_magnetic_field_with_artifacts,
    # Import all the global constants for default values
    BW, VD_GLUCOSE_FACTOR, VD_INSULIN_FACTOR, KA_INSULIN, KE_INSULIN,
    SI_BASE, SI_MULTIPLIER, P2, F01, EGP_B, K_ABS, K_EMP,
    GLUCOSE_KM1, GLUCOSE_KM2, GLUCOSE_KM_ATP, INSULIN_KM_ATP,
    EGP_INSULIN_SENSITIVITY, ATP_BASAL, ATP_MAX, INSULIN_ATP_ENHANCEMENT,
    T1D_MITOCHONDRIAL_EFFICIENCY, HYPOGLYCEMIA_THRESHOLD,
    HYPERGLYCEMIA_THRESHOLD, NORMAL_GLUCOSE_TARGET,
    INSULIN_DOSE_1, INSULIN_DOSE_2, ENSURE_CHO, FASTING_DURATION,
    MAX_SIMULATION_TIME, SIMULATION_DT, N_ATP_SOURCES, FIELD_STRENGTH_BASE,
    INITIAL_GLUCOSE
)

class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic T1D Simulation - ATP-Synthase Magnetic Field Analysis")
        self.root.state('zoomed')

        # Simulation state
        self.simulation_running = False
        self.results = None
        self.magnetic_field_data = None

        # Parameter storage
        self.parameters = {}
        self.setup_parameter_definitions()

        # Create GUI components
        self.setup_gui()

        # Load default parameters
        self.load_default_parameters()

    def setup_parameter_definitions(self):
        """Define all parameter groups and their properties for single tab layout"""
        self.parameter_groups = [
            {
                "name": "Physiological Parameters",
                "color": "#E3F2FD",
                "params": [
                    ("body_weight", BW, "Body Weight (kg)", 40, 120),
                    ("vd_glucose_factor", VD_GLUCOSE_FACTOR, "Glucose Volume Factor (dL/kg)", 1.0, 3.0),
                    ("vd_insulin_factor", VD_INSULIN_FACTOR, "Insulin Volume Factor (L/kg)", 0.05, 0.3),
                ]
            },
            {
                "name": "Insulin Kinetics",
                "color": "#E8F5E8",
                "params": [
                    ("ka_insulin", KA_INSULIN, "Insulin Absorption Rate (min⁻¹)", 0.005, 0.05),
                    ("ke_insulin", KE_INSULIN, "Insulin Elimination Rate (min⁻¹)", 0.01, 0.1),
                    ("si_base", SI_BASE, "Base Insulin Sensitivity", 1e-5, 1e-3),
                    ("si_multiplier", SI_MULTIPLIER, "Insulin Sensitivity Multiplier", 0.3, 2.0),
                    ("p2", P2, "Remote Insulin Rate (min⁻¹)", 0.005, 0.05),
                ]
            },
            {
                "name": "Glucose Metabolism",
                "color": "#FFF3E0",
                "params": [
                    ("f01", F01, "Non-insulin Glucose Use (mg/kg/min)", 0.5, 2.0),
                    ("egp_b", EGP_B, "Endogenous Glucose Prod. (mg/kg/min)", 1.0, 5.0),
                    ("glucose_km1", GLUCOSE_KM1, "Glucose Km1 (mg/dL)", 50, 300),
                    ("glucose_km2", GLUCOSE_KM2, "Glucose Km2 (mg/dL)", 30, 200),
                    ("egp_insulin_sensitivity", EGP_INSULIN_SENSITIVITY, "EGP Insulin Sensitivity", 0.5, 3.0),
                ]
            },
            {
                "name": "Carbohydrates Absorption",
                "color": "#F3E5F5",
                "params": [
                    ("k_abs", K_ABS, "Glucose Absorption Rate (min⁻¹)", 0.01, 0.05),
                    ("k_emp", K_EMP, "Gastric Emptying Rate (min⁻¹)", 0.05, 0.2),
                ]
            },
            {
                "name": "ATP Production",
                "color": "#FFEBEE",
                "params": [
                    ("atp_basal", ATP_BASAL, "Basal ATP Rate (Hz)", 5, 25),
                    ("atp_max", ATP_MAX, "Max ATP Rate (Hz)", 100, 300),
                    ("glucose_km_atp", GLUCOSE_KM_ATP, "Glucose Km for ATP (mg/dL)", 50, 150),
                    ("insulin_km_atp", INSULIN_KM_ATP, "Insulin Km for ATP (μU/mL)", 10, 50),
                    ("insulin_atp_enhancement", INSULIN_ATP_ENHANCEMENT, "Insulin ATP Enhancement", 0.01, 0.1),
                    ("t1d_mitochondrial_efficiency", T1D_MITOCHONDRIAL_EFFICIENCY, "T1D Efficiency", 0.5, 1.0),
                ]
            },
            {
                "name": "Clinical Settings",
                "color": "#E1F5FE",
                "params": [
                    ("hypoglycemia_threshold", HYPOGLYCEMIA_THRESHOLD, "Hypoglycemia Threshold (mg/dL)", 40, 80),
                    ("hyperglycemia_threshold", HYPERGLYCEMIA_THRESHOLD, "Hyperglycemia Threshold (mg/dL)", 150, 250),
                    ("normal_glucose_target", NORMAL_GLUCOSE_TARGET, "Target Glucose (mg/dL)", 80, 120),
                    ("insulin_dose_1", INSULIN_DOSE_1, "First Insulin Dose (units/kg)", 0.01, 0.1),
                    ("insulin_dose_2", INSULIN_DOSE_2, "Second Insulin Dose (units/kg)", 0.02, 0.15),
                    ("ensure_cho", ENSURE_CHO, "Ensure CHO Content (g)", 15, 50),
                ]
            },
            {
                "name": "Simulation & Magnetic Field",
                "color": "#F1F8E9",
                "params": [
                    ("fasting_duration", FASTING_DURATION, "Fasting Duration (min)", 10, 60),
                    ("max_simulation_time", MAX_SIMULATION_TIME, "Max Simulation Time (min)", 200, 600),
                    ("simulation_dt", SIMULATION_DT, "Time Step (min)", 0.1, 1.0),
                    ("n_atp_sources", N_ATP_SOURCES, "Number of ATP Sources", 1000, 10000),
                    ("initial_glucose", INITIAL_GLUCOSE, "Initial Glucose (mg/dL)", 70, 120),
                    ("field_strength_base", FIELD_STRENGTH_BASE, "Base Field Strength (Tesla)", 1e-14, 1e-12),
                    ("sensor_position_x", 50.0, "Sensor X Position (μm)", 0, 100),
                    ("sensor_position_y", 50.0, "Sensor Y Position (μm)", 0, 100),
                    ("sensor_position_z", 10.0, "Sensor Z Position (μm)", 0, 50),
                ]
            },
            {
                "name": "Movement Artifacts & Noise",
                "color": "#FFF8E1",
                "params": [
                    ("heartbeat_amplitude", 2e-13, "Heartbeat Amplitude (Tesla)", 0.5e-13, 10e-13),
                    ("base_heart_rate", 70.0, "Base Heart Rate (BPM)", 50, 120),
                    ("respiratory_amplitude", 1.5e-13, "Respiratory Amplitude (Tesla)", 0.2e-13, 5e-13),
                    ("respiratory_rate", 16.0, "Respiratory Rate (breaths/min)", 8, 30),
                    ("tremor_amplitude", 3e-13, "Tremor Amplitude (Tesla)", 0.5e-13, 10e-13),
                    ("injection_spike_amplitude", 15e-13, "Injection Spike Amplitude (Tesla)", 5e-13, 50e-13),
                    ("muscle_activity_amplitude", 5e-13, "Muscle Activity Amplitude (Tesla)", 1e-13, 20e-13),
                    ("position_drift_amplitude", 5e-13, "Position Drift Amplitude (Tesla)", 1e-13, 20e-13),
                    ("sensor_noise_factor", 0.1, "Sensor Noise Factor", 0.01, 0.5),
                ]
            }
        ]

    def setup_gui(self):
        """Create the main GUI layout with single parameter tab"""
        # Create main paned window
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # Left panel for controls (wider for better parameter visibility)
        left_frame = ttk.Frame(self.main_paned, width=600)
        left_frame.pack_propagate(False)

        # Right panel for plots (larger weight for charts)
        right_frame = ttk.Frame(self.main_paned)

        # Add frames to paned window
        self.main_paned.add(left_frame)
        self.main_paned.add(right_frame)

        self.setup_control_panel(left_frame)
        self.setup_plot_panel(right_frame)

        # Multiple attempts to set the correct sash position
        self.root.update_idletasks()

        # Schedule additional attempts to ensure proper sizing
        self.root.after(50, lambda: self.main_paned.sashpos(0, 600))

    def setup_control_panel(self, parent):
        """Setup the unified parameter control panel"""
        # Create main frame with scrolling
        main_canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )

        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)

        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Title
        title_label = ttk.Label(scrollable_frame, text="Dynamic T1D Simulation Parameters",
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(5, 15))

        self.parameter_widgets = {}

        # Create parameter groups in a grid layout
        for group in self.parameter_groups:
            # Group frame with colored background
            group_frame = tk.Frame(scrollable_frame, bg=group["color"], relief="ridge", bd=2)
            group_frame.pack(fill=tk.X, padx=5, pady=3)

            # Group title
            title_frame = tk.Frame(group_frame, bg=group["color"])
            title_frame.pack(fill=tk.X, padx=5, pady=2)

            ttk.Label(title_frame, text=group["name"], font=("Arial", 11, "bold"),
                     background=group["color"]).pack(anchor=tk.W)

            # Parameters in 2-column layout for better space usage
            params_frame = tk.Frame(group_frame, bg=group["color"])
            params_frame.pack(fill=tk.X, padx=5, pady=2)

            for i, (param_name, default_val, description, min_val, max_val) in enumerate(group["params"]):
                # Determine column (0 or 1)
                col = i % 2
                row = i // 2

                param_frame = tk.Frame(params_frame, bg=group["color"])
                param_frame.grid(row=row, column=col, sticky="ew", padx=3, pady=2)

                # Configure column weights for equal distribution
                params_frame.columnconfigure(0, weight=1)
                params_frame.columnconfigure(1, weight=1)

                # Label
                label = tk.Label(param_frame, text=f"{description}:",
                               font=("Arial", 9), bg=group["color"], anchor="w")
                label.pack(fill=tk.X)

                # Entry and scale frame
                control_frame = tk.Frame(param_frame, bg=group["color"])
                control_frame.pack(fill=tk.X)

                # Entry
                entry = ttk.Entry(control_frame, width=12, font=("Arial", 9))
                entry.pack(side=tk.LEFT, padx=(0, 5))
                entry.insert(0, str(default_val))

                # Scale
                scale = ttk.Scale(control_frame, from_=min_val, to=max_val,
                                orient=tk.HORIZONTAL, length=120)
                scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
                scale.set(default_val)

                # Value display
                value_label = tk.Label(control_frame, text=f"{default_val:.4g}",
                                     font=("Arial", 8), bg=group["color"], width=8)
                value_label.pack(side=tk.RIGHT, padx=(5, 0))

                # Bind entry and scale
                def update_scale(event, s=scale, e=entry, v=value_label):
                    try:
                        val = float(e.get())
                        s.set(val)
                        v.config(text=f"{val:.4g}")
                    except ValueError:
                        pass

                def update_entry(event, s=scale, e=entry, v=value_label):
                    val = s.get()
                    e.delete(0, tk.END)
                    e.insert(0, f"{val:.6g}")
                    v.config(text=f"{val:.4g}")

                entry.bind('<Return>', update_scale)
                entry.bind('<FocusOut>', update_scale)
                scale.bind('<Motion>', update_entry)
                scale.bind('<ButtonRelease-1>', update_entry)

                self.parameter_widgets[param_name] = (entry, scale, value_label)

        # Control buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)

        # First row of buttons
        button_row1 = ttk.Frame(button_frame)
        button_row1.pack(fill=tk.X, pady=2)

        ttk.Button(button_row1, text="🚀 Run Simulation",
                  command=self.run_simulation, width=20).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_row1, text="💾 Save Parameters",
                  command=self.save_parameters, width=20).pack(side=tk.LEFT, padx=2)

        # Second row of buttons
        button_row2 = ttk.Frame(button_frame)
        button_row2.pack(fill=tk.X, pady=2)

        ttk.Button(button_row2, text="📁 Load Parameters",
                  command=self.load_parameters, width=20).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_row2, text="🔄 Reset Defaults",
                  command=self.load_default_parameters, width=20).pack(side=tk.LEFT, padx=2)

        # Progress section
        progress_frame = ttk.Frame(button_frame)
        progress_frame.pack(fill=tk.X, pady=5)

        self.progress_var = tk.StringVar()
        self.progress_var.set("Ready")
        ttk.Label(progress_frame, textvariable=self.progress_var,
                 font=("Arial", 10)).pack()

        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=2)

    def setup_plot_panel(self, parent):
        """Setup the plot display panel to match simulation_dynamic.py"""
        # Create a canvas with scrollbars for the plot area
        plot_canvas = tk.Canvas(parent)
        v_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=plot_canvas.yview)
        h_scrollbar = ttk.Scrollbar(parent, orient="horizontal", command=plot_canvas.xview)
        plot_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        v_scrollbar.pack(side="right", fill="y")
        h_scrollbar.pack(side="bottom", fill="x")
        plot_canvas.pack(side="left", fill="both", expand=True)

        # This frame will contain the plots and will be scrolled
        scrollable_frame = ttk.Frame(plot_canvas)
        plot_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))
        )

        # Create figure with subplots matching the original, with higher DPI
        self.fig = Figure(figsize=(10, 60), dpi=150, facecolor='white')
        self.fig.suptitle('Dynamic T1D Simulation Results', fontsize=16, fontweight='bold')

        # Create canvas and toolbar inside the scrollable frame
        self.canvas = FigureCanvasTkAgg(self.fig, scrollable_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, scrollable_frame)
        toolbar.update()

        ttk.Button(scrollable_frame, text="📊 Export Results",
                  command=self.export_results).pack(side=tk.LEFT, padx=2)
        ttk.Button(scrollable_frame, text="🖼️ Export Plot",
                  command=self.export_plot).pack(side=tk.LEFT, padx=2)

    def get_current_parameters(self):
        """Extract current parameter values from GUI"""
        params = {}
        for param_name, (entry, scale, value_label) in self.parameter_widgets.items():
            try:
                params[param_name] = float(entry.get())
            except ValueError:
                params[param_name] = scale.get()
        return params

    def load_default_parameters(self):
        """Load default parameter values into GUI"""
        for group in self.parameter_groups:
            for param_name, default_val, _, _, _ in group["params"]:
                if param_name in self.parameter_widgets:
                    entry, scale, value_label = self.parameter_widgets[param_name]
                    entry.delete(0, tk.END)
                    entry.insert(0, str(default_val))
                    scale.set(default_val)
                    value_label.config(text=f"{default_val:.4g}")

    def save_parameters(self):
        """Save current parameters to file"""
        params = self.get_current_parameters()
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                json.dump(params, f, indent=2)
            messagebox.showinfo("Success", f"Parameters saved to {filename}")

    def load_parameters(self):
        """Load parameters from file"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    params = json.load(f)

                for param_name, value in params.items():
                    if param_name in self.parameter_widgets:
                        entry, scale, value_label = self.parameter_widgets[param_name]
                        entry.delete(0, tk.END)
                        entry.insert(0, str(value))
                        scale.set(value)
                        value_label.config(text=f"{value:.4g}")

                messagebox.showinfo("Success", f"Parameters loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load parameters: {str(e)}")

    def run_simulation(self):
        """Run the simulation in a separate thread"""
        if self.simulation_running:
            messagebox.showwarning("Warning", "Simulation is already running!")
            return

        # Start simulation in separate thread
        thread = threading.Thread(target=self._run_simulation_thread)
        thread.daemon = True
        thread.start()

    def _run_simulation_thread(self):
        """Run simulation in separate thread"""
        try:
            self.simulation_running = True
            self.progress_bar.start()
            self.progress_var.set("🔄 Running simulation...")

            # Get current parameters
            params = self.get_current_parameters()

            # Create custom simulator with current parameters
            simulator = self.create_custom_simulator(params)

            # Run simulation
            self.progress_var.set("⚙️ Solving differential equations...")
            t_solution, y_solution = simulator.run_simulation()

            # Calculate ATP and magnetic field
            self.progress_var.set("🧬 Calculating ATP production...")
            results = simulator.get_results_dict(t_solution, y_solution)

            self.progress_var.set("🧲 Generating magnetic field signals (please wait)...")
            sources = create_atp_sources(int(params['n_atp_sources']))

            # Get sensor position from GUI parameters
            sensor_position = [
                params['sensor_position_x'],
                params['sensor_position_y'],
                params['sensor_position_z']
            ]

            magnetic_field, biological_signal, movement_artifacts = simulate_magnetic_field_with_artifacts(
                sources, sensor_position, results['atp_rates'], results['time_minutes'],
                results['glucose'], results['insulin'], results['intervention_times'],
                results['interventions']
            )

            # Store results
            self.results = results
            self.magnetic_field_data = {
                'total': magnetic_field,
                'biological': biological_signal,
                'artifacts': movement_artifacts,
                'sources': sources,
                'sensor_position': sensor_position  # Store sensor position for plotting
            }

            # Update plots
            self.progress_var.set("📊 Updating plots...")
            self.root.after(0, self.update_plots)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Simulation failed: {str(e)}"))
        finally:
            self.simulation_running = False
            self.progress_bar.stop()
            self.root.after(0, lambda: self.progress_var.set("✅ Ready"))

    def create_custom_simulator(self, params):
        """Create simulator with custom parameters"""
        simulator = DynamicT1DSimulator()

        # Update simulator parameters
        simulator.BW = params['body_weight']
        simulator.Vd_glucose = params['vd_glucose_factor'] * simulator.BW
        simulator.Vd_insulin = params['vd_insulin_factor'] * simulator.BW
        simulator.ka_insulin = params['ka_insulin']
        simulator.ke_insulin = params['ke_insulin']
        simulator.S_I = params['si_base'] * params['si_multiplier']
        simulator.p2 = params['p2']
        simulator.F01 = params['f01']
        simulator.EGP_b = params['egp_b']
        simulator.k_abs = params['k_abs']
        simulator.k_emp = params['k_emp']
        simulator.HYPOGLYCEMIA_THRESHOLD = params['hypoglycemia_threshold']
        simulator.HYPERGLYCEMIA_THRESHOLD = params['hyperglycemia_threshold']
        simulator.NORMAL_GLUCOSE_TARGET = params['normal_glucose_target']
        simulator.INSULIN_DOSE_1 = params['insulin_dose_1']
        simulator.INSULIN_DOSE_2 = params['insulin_dose_2']
        simulator.ENSURE_CHO = params['ensure_cho']
        simulator.fasting_duration = params['fasting_duration']
        simulator.max_simulation_time = params['max_simulation_time']

        return simulator

    def create_custom_movement_artifacts(self, time_array, glucose, insulin, intervention_times, interventions, params):
        """
        Generate movement artifacts using custom parameters from GUI
        """
        dt = time_array[1] - time_array[0] if len(time_array) > 1 else 1.0
        fs = 1.0 / dt  # Sampling frequency
        artifacts = np.zeros_like(time_array)

        # 1. Heartbeat artifacts with custom parameters
        base_hr = params['base_heart_rate']  # BPM from GUI
        heartbeat_amplitude = params['heartbeat_amplitude']  # Tesla from GUI

        for i, (t, g) in enumerate(zip(time_array, glucose)):
            if g < 70:  # Hypoglycemia - increased heart rate
                hr_factor = 1.5 - 0.5 * (g / 70)  # Up to 50% increase
            elif g > 180:  # Hyperglycemia - moderate increase
                hr_factor = 1.2 + 0.3 * min((g - 180) / 200, 1.0)
            else:
                hr_factor = 1.0

            hr = base_hr * hr_factor
            heartbeat_freq = hr / 60  # Convert to Hz

            # Heartbeat signal with harmonics using custom amplitude
            heartbeat = (heartbeat_amplitude * np.sin(2 * np.pi * heartbeat_freq * t) +
                        heartbeat_amplitude * 0.5 * np.sin(2 * np.pi * 2 * heartbeat_freq * t) +
                        heartbeat_amplitude * 0.25 * np.sin(2 * np.pi * 3 * heartbeat_freq * t))
            artifacts[i] += heartbeat

        # 2. Respiratory artifacts with custom parameters
        respiratory_rate = params['respiratory_rate'] / 60  # Convert to Hz
        respiratory_amplitude = params['respiratory_amplitude']  # Tesla from GUI
        respiratory_artifacts = respiratory_amplitude * np.sin(2 * np.pi * respiratory_rate * time_array)
        artifacts += respiratory_artifacts

        # 3. Tremor artifacts during hyperglycemia with custom amplitude
        tremor_amplitude = params['tremor_amplitude']  # Tesla from GUI
        tremor_artifacts = np.zeros_like(time_array)
        for i, g in enumerate(glucose):
            if g > 200:  # Hyperglycemic tremors
                tremor_intensity = min((g - 200) / 100, 1.0) * tremor_amplitude
                tremor_freq = 6.0  # Hz
                tremor_artifacts[i] = tremor_intensity * np.sin(2 * np.pi * tremor_freq * time_array[i])
        artifacts += tremor_artifacts

        # 4. Injection movement artifacts with custom amplitude
        injection_amplitude = params['injection_spike_amplitude']  # Tesla from GUI
        for intervention in interventions:
            if intervention.event_type in ['insulin_1', 'insulin_2']:
                # Find closest time index
                t_intervention = intervention.time * 60  # Convert to seconds
                idx = np.argmin(np.abs(time_array - t_intervention))

                # Injection artifact: sharp spike followed by exponential decay
                artifact_duration = 30  # seconds
                start_idx = max(0, idx - 5)
                end_idx = min(len(time_array), idx + int(artifact_duration / dt))

                if end_idx > start_idx:
                    t_rel = time_array[start_idx:end_idx] - time_array[idx]
                    injection_artifact = np.zeros(end_idx - start_idx)

                    # Sharp spike at injection with custom amplitude
                    spike_mask = np.abs(t_rel) < 2  # 2-second spike
                    injection_artifact[spike_mask] = injection_amplitude * np.exp(-np.abs(t_rel[spike_mask]))

                    # Post-injection movement (patient discomfort)
                    post_mask = (t_rel > 0) & (t_rel < 20)
                    injection_artifact[post_mask] += injection_amplitude * 0.33 * np.exp(-t_rel[post_mask] / 10) * np.sin(2 * np.pi * 0.5 * t_rel[post_mask])

                    artifacts[start_idx:end_idx] += injection_artifact

        # 5. Muscular activity artifacts with custom amplitude
        muscle_amplitude_base = params['muscle_activity_amplitude']  # Tesla from GUI
        for i, g in enumerate(glucose):
            if np.random.random() < 0.001:  # Random muscle activity
                activity_factor = 2.0 if g < 70 else 1.0  # Increased during hypoglycemia
                muscle_duration = int(np.random.uniform(1, 5) / dt)  # 1-5 seconds

                start_idx = i
                end_idx = min(len(time_array), i + muscle_duration)

                if end_idx > start_idx:
                    muscle_freq = np.random.uniform(15, 40)  # Hz
                    muscle_amplitude = activity_factor * muscle_amplitude_base * np.random.uniform(0.4, 1.6)
                    t_muscle = time_array[start_idx:end_idx] - time_array[start_idx]
                    muscle_signal = muscle_amplitude * np.exp(-t_muscle / 2) * np.sin(2 * np.pi * muscle_freq * t_muscle)
                    artifacts[start_idx:end_idx] += muscle_signal

        # 6. Low-frequency drift from patient positioning with custom amplitude
        position_drift_amplitude = params['position_drift_amplitude']  # Tesla from GUI
        position_drift = position_drift_amplitude * np.sin(2 * np.pi * 0.05 * time_array) * np.exp(-time_array / 3600)
        artifacts += position_drift

        return artifacts

    def add_glycemic_range_coloring(self, ax, time_data, glucose_data):
        """Add colored background for hypoglycemic and hyperglycemic ranges"""
        HYPOGLYCEMIA_THRESHOLD = 70.0
        HYPERGLYCEMIA_THRESHOLD = 180.0

        hypo_mask = glucose_data < HYPOGLYCEMIA_THRESHOLD
        hyper_mask = glucose_data > HYPERGLYCEMIA_THRESHOLD

        if np.any(hypo_mask):
            y_min, y_max = ax.get_ylim()
            ax.fill_between(time_data, y_min, y_max, where=hypo_mask,
                           color='blue', alpha=0.1, label='Hypoglycemic range')

        if np.any(hyper_mask):
            y_min, y_max = ax.get_ylim()
            ax.fill_between(time_data, y_min, y_max, where=hyper_mask,
                           color='red', alpha=0.1, label='Hyperglycemic range')

    def update_plots(self):
        """Update the plot panel with simulation results to match simulation_dynamic.py exactly"""
        if self.results is None:
            return

        # Clear previous plots
        self.fig.clear()

        # Import scipy.signal for filtering (matching original)
        from scipy import signal

        # Extract data
        time_min = self.results['time_minutes']
        glucose = self.results['glucose']
        insulin = self.results['insulin']
        atp_rates = self.results['atp_rates']
        magnetic_field = self.magnetic_field_data['total']
        biological_signal = self.magnetic_field_data['biological']
        movement_artifacts = self.magnetic_field_data['artifacts']
        intervention_times = self.results['intervention_times']

        # Phase information - exactly matching original
        phase_times = [0] + intervention_times + [time_min[-1]]
        phase_names = ['Fasting', 'Insulin-1', 'Ensure', 'Insulin-2']
        phase_colors = ['blue', 'red', 'green', 'purple']

        # Define glycemic ranges for coloring - matching original
        HYPOGLYCEMIA_THRESHOLD = 70.0  # mg/dL
        HYPERGLYCEMIA_THRESHOLD = 180.0  # mg/dL

        # Helper function to add glycemic range coloring - exactly matching original
        def add_glycemic_range_coloring(ax, time_data, glucose_data):
            """Add colored background for hypoglycemic and hyperglycemic ranges"""
            # Create boolean masks for glycemic ranges
            hypo_mask = glucose_data < HYPOGLYCEMIA_THRESHOLD
            hyper_mask = glucose_data > HYPERGLYCEMIA_THRESHOLD
            normal_mask = (glucose_data >= HYPOGLYCEMIA_THRESHOLD) & (glucose_data <= HYPERGLYCEMIA_THRESHOLD)

            # Find continuous regions
            if np.any(hypo_mask):
                y_min, y_max = ax.get_ylim()
                ax.fill_between(time_data, y_min, y_max, where=hypo_mask,
                               color='blue', alpha=0.1, label='Hypoglycemic range')

            if np.any(hyper_mask):
                y_min, y_max = ax.get_ylim()
                ax.fill_between(time_data, y_min, y_max, where=hyper_mask,
                               color='red', alpha=0.1, label='Hyperglycemic range')

            if np.any(normal_mask):
                y_min, y_max = ax.get_ylim()
                ax.fill_between(time_data, y_min, y_max, where=normal_mask,
                               color='green', alpha=0.05, label='Normal range')

        # 1. Glucose levels over time - exactly matching original
        ax1 = self.fig.add_subplot(7, 1, 1)
        ax1.plot(time_min, glucose, 'b-', linewidth=2, label='Glucose')
        ax1.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Hypoglycemia (70)')
        ax1.axhline(y=180, color='orange', linestyle='--', alpha=0.7, label='Hyperglycemia (180)')
        ax1.axhline(y=90, color='g', linestyle='--', alpha=0.7, label='Target (90)')

        # Mark interventions with labels first
        for i, t_int in enumerate(intervention_times):
            if i < len(phase_names):
                ax1.axvline(x=t_int, color=phase_colors[i+1], linestyle=':', alpha=0.8)

        # Set labels and formatting before adding background coloring
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('Glucose (mg/dL)')
        ax1.set_title('Dynamic Glucose Response')
        ax1.grid(True, alpha=0.3)

        # Add glycemic range coloring after axis is properly set up
        add_glycemic_range_coloring(ax1, time_min, glucose)

        # Add intervention labels after background coloring
        for i, t_int in enumerate(intervention_times):
            if i < len(phase_names):
                ax1.text(t_int, ax1.get_ylim()[1]*0.95, phase_names[i+1],
                        rotation=90, verticalalignment='top', color=phase_colors[i+1])

        ax1.legend(fontsize=8)

        # 2. Insulin levels - exactly matching original
        ax2 = self.fig.add_subplot(7, 1, 2)
        ax2.plot(time_min, insulin, 'g-', linewidth=2)

        # Mark interventions first
        for i, t_int in enumerate(intervention_times):
            ax2.axvline(x=t_int, color=phase_colors[i+1], linestyle=':', alpha=0.8)

        ax2.set_xlabel('Time (min)')
        ax2.set_ylabel('Insulin (μU/mL)')
        ax2.set_title('Insulin Concentration')
        ax2.grid(True, alpha=0.3)

        # Add glycemic range coloring after axis setup
        add_glycemic_range_coloring(ax2, time_min, glucose)

        # 3. ATP production rates - exactly matching original
        ax3 = self.fig.add_subplot(7, 1, 3)
        ax3.plot(time_min, atp_rates, 'purple', linewidth=2)

        # Mark interventions first
        for i, t_int in enumerate(intervention_times):
            ax3.axvline(x=t_int, color=phase_colors[i+1], linestyle=':', alpha=0.8)

        ax3.set_xlabel('Time (min)')
        ax3.set_ylabel('ATP Rate (Hz)')
        ax3.set_title('ATP Production Rate')
        ax3.grid(True, alpha=0.3)

        # Add glycemic range coloring after axis setup
        add_glycemic_range_coloring(ax3, time_min, glucose)

        # 4. Enhanced Magnetic field signal with low-frequency overlay - exactly matching original
        ax4 = self.fig.add_subplot(7, 1, 4)

        # Convert to pT for display
        magnetic_field_pT = magnetic_field * 1e12
        biological_signal_pT = biological_signal * 1e12
        movement_artifacts_pT = movement_artifacts * 1e12

        # Create low-frequency overlay using 10 Hz cutoff filter - exactly matching original
        dt = time_min[1] - time_min[0] if len(time_min) > 1 else 0.1  # minutes
        fs = 1.0 / (dt * 60)  # Convert to Hz

        # Apply low-pass filter (5 Hz cutoff for clear trend visualization)
        try:
            sos = signal.butter(4, 5.0, btype='low', fs=fs, output='sos')
            magnetic_field_filtered = signal.sosfilt(sos, magnetic_field_pT)
            biological_filtered = signal.sosfilt(sos, biological_signal_pT)
        except:
            # Fallback if filtering fails
            magnetic_field_filtered = magnetic_field_pT
            biological_filtered = biological_signal_pT

        # Plot high-frequency signals with transparency - exactly matching original
        ax4.plot(time_min, magnetic_field_pT, 'k-', linewidth=0.5, alpha=0.3, label='Total (raw)')
        ax4.plot(time_min, biological_signal_pT, 'r-', linewidth=0.3, alpha=0.4, label='Biological (raw)')
        ax4.plot(time_min, movement_artifacts_pT, 'orange', linewidth=0.3, alpha=0.4, label='Movement artifacts')

        # Overlay low-frequency trends
        ax4.plot(time_min, magnetic_field_filtered, 'black', linewidth=2, alpha=0.9, label='Total (5Hz low-pass)')
        ax4.plot(time_min, biological_filtered, 'darkred', linewidth=2, alpha=0.8, label='Biological (5Hz low-pass)')

        # Add glycemic range coloring
        add_glycemic_range_coloring(ax4, time_min, glucose)

        # Mark interventions
        for i, t_int in enumerate(intervention_times):
            ax4.axvline(x=t_int, color=phase_colors[i+1], linestyle=':', alpha=0.8)

        ax4.set_xlabel('Time (min)')
        ax4.set_ylabel('Magnetic Field (pT)')
        ax4.set_title('Magnetic Field Signal with Low-Frequency Overlay')
        ax4.legend(fontsize=8, loc='upper right')
        ax4.grid(True, alpha=0.3)

        # 5. Biological signal (low pass) vs time - exactly matching original
        ax5 = self.fig.add_subplot(7, 1, 5)

        # Apply low-pass filter to biological signal for trend visualization
        try:
            sos = signal.butter(4, 5.0, btype='low', fs=fs, output='sos')
            biological_filtered = signal.sosfilt(sos, biological_signal_pT)
        except:
            # Fallback if filtering fails
            biological_filtered = biological_signal_pT

        # Calculate envelope of the biological signal using Hilbert transform
        try:
            from scipy.signal import hilbert
            from scipy.ndimage import uniform_filter1d

            # Calculate envelope for raw biological signal
            analytic_signal = hilbert(biological_signal_pT)
            envelope_raw = np.abs(analytic_signal)

            # Calculate envelope for filtered biological signal
            analytic_signal_filtered = hilbert(biological_filtered)
            envelope_filtered = np.abs(analytic_signal_filtered)

            # Smooth the envelopes for cleaner visualization
            # Use a smoothing window that's about 1% of the signal length
            smooth_window = max(5, len(envelope_raw) // 100)
            envelope_raw_smooth = uniform_filter1d(envelope_raw, size=smooth_window)
            envelope_filtered_smooth = uniform_filter1d(envelope_filtered, size=smooth_window)

            # Plot smooth envelope with transparency
            ax5.fill_between(time_min, envelope_raw_smooth, -envelope_raw_smooth,
                           color='lightcoral', alpha=0.15, label='Raw signal envelope (smooth)')
            ax5.fill_between(time_min, envelope_filtered_smooth, -envelope_filtered_smooth,
                           color='darkred', alpha=0.2, label='Filtered signal envelope (smooth)')

        except:
            # If Hilbert transform fails, use a simpler moving maximum/minimum envelope with smoothing
            from scipy.ndimage import uniform_filter1d
            window_size = max(10, len(biological_signal_pT) // 50)  # Larger window for smoother envelope

            # Calculate rolling max and min for envelope
            envelope_max = uniform_filter1d(np.maximum.accumulate(biological_signal_pT), size=window_size)
            envelope_min = uniform_filter1d(np.minimum.accumulate(biological_signal_pT), size=window_size)

            ax5.fill_between(time_min, envelope_max, envelope_min,
                           color='lightcoral', alpha=0.15, label='Signal envelope (smooth)')

        # Plot both raw and filtered biological signals - exactly matching original
        ax5.plot(time_min, biological_signal_pT, 'lightcoral', linewidth=0.5, alpha=0.4, label='Raw biological')
        ax5.plot(time_min, biological_filtered, 'darkred', linewidth=2, alpha=0.9, label='Low-pass filtered (5Hz)')

        # Plot envelope boundary lines in black on top for visibility
        try:
            # Plot the envelope boundary lines in black for better visibility
            ax5.plot(time_min, envelope_raw_smooth, 'black', linewidth=1.5, alpha=0.8, linestyle='--', label='Raw envelope')
            ax5.plot(time_min, -envelope_raw_smooth, 'black', linewidth=1.5, alpha=0.8, linestyle='--')
            ax5.plot(time_min, envelope_filtered_smooth, 'black', linewidth=2, alpha=0.9, linestyle='-', label='Filtered envelope')
            ax5.plot(time_min, -envelope_filtered_smooth, 'black', linewidth=2, alpha=0.9, linestyle='-')
        except:
            # Fallback envelope lines
            ax5.plot(time_min, envelope_max, 'black', linewidth=2, alpha=0.8, linestyle='-', label='Envelope bounds')
            ax5.plot(time_min, envelope_min, 'black', linewidth=2, alpha=0.8, linestyle='-')

        # Add glycemic range coloring
        add_glycemic_range_coloring(ax5, time_min, glucose)

        # Mark interventions
        for i, t_int in enumerate(intervention_times):
            ax5.axvline(x=t_int, color=phase_colors[i+1], linestyle=':', alpha=0.8)

        ax5.set_xlabel('Time (min)')
        ax5.set_ylabel('Magnetic Field (pT)')
        ax5.set_title('Biological Signal (Low Pass) vs Time')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        # 6. Enhanced Spectrogram with dB/Hz scale - exactly matching original
        ax6 = self.fig.add_subplot(7, 1, 6)
        if len(magnetic_field) > 50:
            # Calculate spectrogram with improved parameters for better resolution
            f, t_spec, Sxx = spectrogram(magnetic_field, fs=fs,
                                       nperseg=min(256, len(magnetic_field)//4),
                                       noverlap=min(128, len(magnetic_field)//8),
                                       window='hann')

            # Convert to dB/Hz scale
            # PSD in T²/Hz, convert to dB relative to 1 T²/Hz
            Sxx_dB_Hz = 10 * np.log10(Sxx + 1e-30)  # Avoid log(0)

            # Convert time to minutes for x-axis
            t_spec_min = t_spec / 60

            im = ax6.pcolormesh(t_spec_min, f, Sxx_dB_Hz, shading='gouraud', cmap='viridis')
            cbar = self.fig.colorbar(im, ax=ax6, label='PSD (dB/Hz re 1 T²/Hz)')

            # Mark interventions
            for i, t_int in enumerate(intervention_times):
                ax6.axvline(x=t_int, color=phase_colors[i+1], linestyle=':', alpha=0.8)

            ax6.set_ylabel('Frequency (Hz)')
            ax6.set_xlabel('Time (min)')
            ax6.set_title('Spectrogram (dB/Hz Scale)')
            ax6.set_ylim(0, min(50, fs/2))  # Limit to 50 Hz or Nyquist frequency

        # 7. 3D source distribution with improved visualization - exactly matching original
        ax7 = self.fig.add_subplot(7, 1, 7, projection='3d')
        sources = self.magnetic_field_data['sources']
        positions = np.array([s.position for s in sources])
        activities = [s.metabolic_activity for s in sources]

        scatter = ax7.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                             c=activities, cmap='viridis', s=30, alpha=0.7)

        # Get actual sensor position from GUI parameters
        actual_sensor_position = self.magnetic_field_data['sensor_position']
        tissue_center = [0, 0, 25]  # Center of tissue volume

        # Create cylindrical sensor (typical magnetic sensor dimensions)
        # Orient sensor so the sensing surface faces toward the tissue center
        sensor_radius = 3.0  # μm
        sensor_height = 8.0  # μm

        # Add sensor marker showing actual position from GUI
        ax7.scatter([actual_sensor_position[0]], [actual_sensor_position[1]], [actual_sensor_position[2]],
                   color='red', s=100, marker='s',
                   label=f'Sensor @ ({actual_sensor_position[0]:.1f}, {actual_sensor_position[1]:.1f}, {actual_sensor_position[2]:.1f}) μm')

        ax7.set_xlabel('X (μm)')
        ax7.set_ylabel('Y (μm)')
        ax7.set_zlabel('Z (μm)')
        ax7.set_title('ATP-Synthase Distribution with Magnetic Sensor')

        # Set equal aspect ratio for better visualization
        ax7.set_box_aspect([1,1,0.5])  # Make Z-axis shorter for better view

        self.fig.colorbar(scatter, ax=ax7, label='Metabolic Activity', shrink=0.8)
        ax7.legend(loc='upper left', fontsize=8)

        # Apply tight layout and update canvas - exactly matching original
        self.fig.tight_layout()
        self.canvas.draw()

    def export_results(self):
        """Export simulation results to file"""
        if self.results is None:
            messagebox.showwarning("Warning", "No simulation results to export!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                if filename.endswith('.json'):
                    # Export as JSON
                    export_data = {
                        'parameters': self.get_current_parameters(),
                        'results': {
                            'time_minutes': self.results['time_minutes'].tolist(),
                            'glucose': self.results['glucose'].tolist(),
                            'insulin': self.results['insulin'].tolist(),
                            'atp_rates': self.results['atp_rates'].tolist(),
                            'magnetic_field': self.magnetic_field_data['total'].tolist(),
                            'biological_signal': self.magnetic_field_data['biological'].tolist(),
                            'movement_artifacts': self.magnetic_field_data['artifacts'].tolist(),
                        },
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(filename, 'w') as f:
                        json.dump(export_data, f, indent=2)

                elif filename.endswith('.csv'):
                    # Export as CSV
                    import pandas as pd
                    df = pd.DataFrame({
                        'time_minutes': self.results['time_minutes'],
                        'glucose': self.results['glucose'],
                        'insulin': self.results['insulin'],
                        'atp_rates': self.results['atp_rates'],
                        'magnetic_field': self.magnetic_field_data['total'],
                        'biological_signal': self.magnetic_field_data['biological'],
                        'movement_artifacts': self.magnetic_field_data['artifacts'],
                    })
                    df.to_csv(filename, index=False)

                messagebox.showinfo("Success", f"Results exported to {filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")

    def export_plot(self):
        """Export current plot to file"""
        if self.results is None:
            messagebox.showwarning("Warning", "No plot to export!")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"),
                      ("SVG files", "*.svg"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export plot: {str(e)}")


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = SimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
