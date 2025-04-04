import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
import re

# Define exponential decay model
def exp_decay(x, D0, mu):
    return D0 * np.exp(-mu * x)


def get_material_data(base_path="./Data"):
    """
    Scans each subdirectory under ./Data/ and loads file paths and thicknesses.
    Assumes file names contain the thickness in the format '100MeV[THICKNESS]cm'
    """
    materials = {}

    for material_name in os.listdir(base_path):
        material_path = os.path.join(base_path, material_name)
        if not os.path.isdir(material_path):
            continue  # Skip files in the root ./Data directory

        file_paths = []
        thicknesses = []

        for fname in os.listdir(material_path):
            if not fname.endswith(".txt"):
                continue
            
            match = re.search(r'(\d+)cm', fname)
            if match:
                thickness = int(match.group(1))
                full_path = os.path.join(material_path, fname)
                file_paths.append(full_path)
                thicknesses.append(thickness)
        
        # Sort by thickness
        sorted_data = sorted(zip(thicknesses, file_paths))
        if not sorted_data:
            print(f"⚠️ No valid data found for material: {material_name}")
            continue  # Skip this material
        thicknesses, file_paths = zip(*sorted_data)

        materials[material_name] = {
            "file_paths": list(file_paths),
            "thicknesses": list(thicknesses)
        }

    return materials

def extract_dose_and_errors(file_path):
    """
    Reads a .txt file containing dose values and percentage errors from FLUKA output.
    Returns two NumPy arrays: one with the dose values and one with the percentage errors.
    """
    dose_values = []
    error_values = []
    in_dose_section = False
    in_error_section = False
    
    with open(file_path, "r") as file:
        lines = file.readlines()
        
    for line in lines:
        if "Data follow in a matrix" in line:
            in_dose_section = True
            in_error_section = False
            continue
        if "Percentage errors follow in a matrix" in line:
            in_dose_section = False
            in_error_section = True
            continue
        
        if in_dose_section:
            try:
                numbers = [float(x) for x in line.split()]
                dose_values.extend(numbers)
            except ValueError:
                continue
        elif in_error_section:
            try:
                numbers = [float(x) for x in line.split()]
                error_values.extend(numbers)
            except ValueError:
                continue
                
    return np.array(dose_values), np.array(error_values)

# Container for final results
absorption_data = []

# Define materials and their data files
materials = get_material_data("./Data")


# Function to extract and fit data per material
def process_material(material_name, file_paths, thicknesses):
    results = []
    for file_path, thickness in zip(file_paths, thicknesses):
        dose_values, error_values = extract_dose_and_errors(file_path)
        mean_dose = np.mean(dose_values)
        normalised_dose = mean_dose * 1e6
        total_error = np.sum(error_values)
        normalised_error = total_error / 1000
        error_in_dose = normalised_dose * (normalised_error / 100.0)
        results.append([thickness, normalised_dose, error_in_dose])
    
    df = pd.DataFrame(results, columns=["Thickness", "Dose", "AbsError"])

    # Remove NaNs
    df = df.dropna()

    # Fit exponential
    x = df["Thickness"].values
    y = df["Dose"].values
    yerr = df["AbsError"].values

    popt, pcov = curve_fit(
        exp_decay, x, y, 
        sigma=yerr, 
        absolute_sigma=True, 
        p0=(y[0], 0.1), 
        bounds=([0, 0], 
        [np.inf, np.inf])  # D0 ≥ 0 and μ ≥ 0)
    )
    D0_fit, mu_fit = popt
    D0_err, mu_err = np.sqrt(np.diag(pcov))

    # Store result
    absorption_data.append({
        "Material": material_name,
        "D0 (pSv)": D0_fit,
        "μ (cm⁻¹)": mu_fit,
        "μ Error": mu_err
    })

# Run through all materials
for material, data in materials.items():
    process_material(material, data["file_paths"], data["thicknesses"])

# Save results to CSV or PDF
absorption_df = pd.DataFrame(absorption_data)
absorption_df = absorption_df.round(4)
absorption_df.to_csv("absorption_coefficients_summary.csv", index=False)
print(absorption_df)
