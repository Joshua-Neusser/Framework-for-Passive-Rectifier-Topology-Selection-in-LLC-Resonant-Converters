import fitz
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from tkinter import Tk, filedialog
import os

g_points = []
def select_file(title="Select the file"):
    Tk().withdraw(); return filedialog.askopenfilename(title=title)
def onclick(event):
    global g_points
    if event.xdata is not None and event.ydata is not None:
        g_points.append((event.xdata, event.ydata)); print(f"Calibration point {len(g_points)} captured.")
        plt.plot(event.xdata, event.ydata, 'r+', markersize=10); plt.draw()
        if len(g_points) == 4: plt.close()
def get_axis_calibration(image_path):
    global g_points; g_points = []
    img = cv2.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 8)); ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title("Calibration: Click on the 4 points in the indicated order")
    fig.canvas.mpl_connect('button_press_event', onclick)
    print("\n--- AXIS CALIBRATION INSTRUCTIONS ---\n1. Click on X MIN\n2. Click on X MAX\n3. Click on Y MIN (bottom)\n4. Click on Y MAX (top)")
    plt.show()
    if len(g_points) != 4: raise Exception("Calibration canceled.")
    vals = [float(input(f"Enter the value for point {i+1}: ")) for i in range(4)]
    p_x_min, _ = g_points[0]; p_x_max, _ = g_points[1]; _, y_click1 = g_points[2]; _, y_click2 = g_points[3]
    return {'px_min': p_x_min, 'px_max': p_x_max, 'py_min': max(y_click1, y_click2), 'py_max': min(y_click1, y_click2),
            'val_x_min': vals[0], 'val_x_max': vals[1], 'val_y_min': vals[2], 'val_y_max': vals[3]}
def pixel_to_data(px, py, cal):
    x_ratio = (px - cal['px_min']) / (cal['px_max'] - cal['px_min'])
    data_x = cal['val_x_min'] + x_ratio * (cal['val_x_max'] - cal['val_x_min'])
    y_ratio = (cal['py_min'] - py) / (cal['py_min'] - cal['py_max'])
    log_y_min, log_y_max = np.log10(cal['val_y_min']), np.log10(cal['val_y_max'])
    data_y = 10**(log_y_min + y_ratio * (log_y_max - log_y_min))
    return data_x, data_y

def diode_surface_model_deg3(x_data, k_c3, d_c3, k_c2, d_c2, k_c1, d_c1, k_c0, d_c0):
    """
    Calculates Vf as a function of Current (I) and Temperature (T).
    Model: Vf(I, T) = c3(T)*ln(I)^3 + c2(T)*ln(I)^2 + c1(T)*ln(I) + c0(T)
            where c_i(T) = k_ci*T + d_ci
    """
    current, temp = x_data
    log_i = np.log(current)
    
    c3 = k_c3 * temp + d_c3
    c2 = k_c2 * temp + d_c2
    c1 = k_c1 * temp + d_c1
    c0 = k_c0 * temp + d_c0
    
    return c3 * (log_i**3) + c2 * (log_i**2) + c1 * log_i + c0

# --- MAIN FUNCTION ---
def main():
    image_path = select_file("Select the graph image")
    if not image_path: return

    try:
        calibration = get_axis_calibration(image_path)
    except Exception as e:
        print(f"Error during calibration: {e}"); return

    all_raw_data = []
    
    # STEP 1: Interactive curve extraction
    while True:
        curve_count = len(all_raw_data) + 1
        img = cv2.imread(image_path)
        fig, ax = plt.subplots(figsize=(12, 9)); ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"CURVE {curve_count}: Click on the points. Press ENTER to finish.")
        
        print(f"\n--- EXTRACTING CURVE {curve_count} ---")
        clicked_points = plt.ginput(n=-1, timeout=0, show_clicks=True)
        plt.close()
        
        if clicked_points:
            temp = float(input(f"Enter the temperature (°C) for Curve {curve_count}: "))
            raw_data = pd.DataFrame([pixel_to_data(px, py, calibration) for px, py in clicked_points],
                                    columns=['Voltage (V)', 'Current (A)'])
            raw_data['Temperature (C)'] = temp
            all_raw_data.append(raw_data)

        if input("\nDo you want to extract another curve? (y/n): ").lower() != 'y':
            break
            
    if len(all_raw_data) < 2:
        print("\nError: At least two temperature curves are required for model fitting."); return
        
    # STEP 2: Consolidation and model fitting
    print("\n--------------------------------------------------")
    print("Processing data... Curve fitting...")
    
    full_df = pd.concat(all_raw_data, ignore_index=True)
    x_data = [full_df['Current (A)'], full_df['Temperature (C)']]; y_data = full_df['Voltage (V)']
    
    try:
        # Initial guesses for the 8 coefficients
        initial_guesses = [0, 0, 0, 0, 0, 0.1, 0, 1]
        params, _ = curve_fit(diode_surface_model_deg3, x_data, y_data, p0=initial_guesses)
        
        # STEP 3: Display results
        print("\n--- FIT SUCCESSFULLY COMPLETED! ---")
        print("\nThe final 3rd-degree polynomial equation for the diode is:")
        print("Vf(I, T) = c3(T)*ln(I)³ + c2(T)*ln(I)² + c1(T)*ln(I) + c0(T)")
        print("         where c_i(T) = k_ci*T + d_ci\n")
        print("Copy the 8 coefficients below to your database:")
        print("--------------------------------------------------")
        print(f"  k_c3 = {params[0]:.8f}")
        print(f"  d_c3 = {params[1]:.8f}")
        print(f"  k_c2 = {params[2]:.8f}")
        print(f"  d_c2 = {params[3]:.8f}")
        print(f"  k_c1 = {params[4]:.8f}")
        print(f"  d_c1 = {params[5]:.8f}")
        print(f"  k_c0 = {params[6]:.8f}")
        print(f"  d_c0 = {params[7]:.8f}")
        print("--------------------------------------------------")

        # Save coefficients to a text file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        txt_path = os.path.join(script_dir, "final_diode_coefficients_deg3.txt")
        with open(txt_path, "w") as f:
            f.write("Coefficients for the 3rd-degree polynomial equation:\n")
            f.write("Vf(I, T) = (k_c3*T + d_c3)*ln(I)³ + (k_c2*T + d_c2)*ln(I)² + (k_c1*T + d_c1)*ln(I) + (k_c0*T + d_c0)\n")
            f.write("--------------------------------------------------\n")
            f.write(f"k_c3 = {params[0]:.8f}\n"); f.write(f"d_c3 = {params[1]:.8f}\n")
            f.write(f"k_c2 = {params[2]:.8f}\n"); f.write(f"d_c2 = {params[3]:.8f}\n")
            f.write(f"k_c1 = {params[4]:.8f}\n"); f.write(f"d_c1 = {params[5]:.8f}\n")
            f.write(f"k_c0 = {params[6]:.8f}\n"); f.write(f"d_c0 = {params[7]:.8f}\n")
        print(f"\nCoefficients also saved in: {txt_path}")

        # STEP 4: Visual verification
        plt.figure(figsize=(10, 7))
        unique_temps = full_df['Temperature (C)'].unique()
        for temp in sorted(unique_temps):
            subset_raw = full_df[full_df['Temperature (C)'] == temp]
            plt.plot(subset_raw['Voltage (V)'], subset_raw['Current (A)'], 'o', label=f'Original Points {temp}°C')
            current_range = np.logspace(np.log10(full_df['Current (A)'].min()), np.log10(full_df['Current (A)'].max()), 200)
            temp_range = np.full_like(current_range, temp)
            voltage_fit = diode_surface_model_deg3([current_range, temp_range], *params)
            plt.plot(voltage_fit, current_range, '-', label=f'Final Equation {temp}°C')
            
        plt.yscale('log'); plt.grid(True, which="both", ls="--")
        plt.title("Verification: Original Points vs. Fitted Curve")
        plt.xlabel("Forward Voltage (V)", fontsize=16); plt.ylabel("Forward Current (A)", fontsize=16)
        plt.xticks(fontsize=16) 
        plt.yticks(fontsize=16) 
        plt.legend(); plt.show()

    except RuntimeError:
        print("\nError: Could not converge. Try extracting more points or check input data.")

if __name__ == '__main__':
    main()
