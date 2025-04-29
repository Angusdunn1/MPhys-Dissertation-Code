import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

# Define the results directory and ensure it exists
results_dir = "./Results/Aluminium/"
os.makedirs(results_dir, exist_ok=True)

# Manually specify the data file paths and corresponding thickness values
file_paths = [
    "./Data/Aluminium/100MeV3.5cm_51.txt",
    "./Data/Aluminium/100MeV4cm_51.txt",
    "./Data/Aluminium/100MeV4.5_51.txt",
    "./Data/Aluminium/100MeV6cm_51.txt",
    "./Data/Aluminium/100MeV8cm_51.txt",
    "./Data/Aluminium/100MeV10cm_51.txt"
]

thicknesses = [3.5, 4, 4.5, 6, 8, 10]  # Corresponding thickness values in cm

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

# Process only the selected files
results = []

for file_path, thickness in zip(file_paths, thicknesses):
    dose_values, error_values = extract_dose_and_errors(file_path)
    
    mean_dose = np.mean(dose_values)
    normalised_dose = mean_dose * 1e6  # Convert to pSv

    total_error = np.sum(error_values)
    normalised_error = total_error / 1000  # Normalise percentage error per bin
    error_in_dose = normalised_dose * (normalised_error / 100.0)  # Absolute error in pSv

    results.append([thickness, normalised_dose, normalised_error, error_in_dose])

# Convert results into a Pandas DataFrame
df_zoomed = pd.DataFrame(results, columns=["Thickness (cm)", "Dose per Proton (pSv)", "Normalised Error (%)", "Absolute Error (pSv)"])

# Round for scientific readability
df_zoomed = df_zoomed.round({
    "Thickness (cm)": 2,
    "Dose per Proton (pSv)": 3,
    "Normalised Error (%)": 2,
    "Absolute Error (pSv)": 3
})

# Ensure no zero values for plotting
df_zoomed["Dose per Proton (pSv)"] = df_zoomed["Dose per Proton (pSv)"].replace(0, pd.NA)

################################################################################
# **Generate and Save the Plot**
################################################################################

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_zoomed["Thickness (cm)"],
    y=df_zoomed["Dose per Proton (pSv)"],
    error_y=dict(type='data', array=df_zoomed["Absolute Error (pSv)"], visible=True),
    mode='lines+markers',
    marker=dict(size=10, color='black', symbol='circle'),  # Black markers for professional look
    line=dict(width=3, color='black'),  # Black line for clean academic style
    name="Dose per Proton (Zoomed)"
))

# Improve layout for the zoomed-in scientific plot
fig.update_layout(
    title="Dose per Proton vs. Thickness (Zoomed In: Thickness > 3.5 cm)",
    title_font=dict(size=20, family="Arial"),
    xaxis=dict(
        title="Material Thickness (cm)",
        title_font=dict(size=18),
        tickfont=dict(size=14),
        showgrid=True
    ),
    yaxis=dict(
        title="Dose per Proton (pSv)",  # Linear scale (not log)
        title_font=dict(size=18),
        tickfont=dict(size=14),
        showgrid=True
    ),
    template="simple_white",
    hovermode="x",
    font=dict(family="Arial", size=14),
    margin=dict(l=80, r=80, t=50, b=50)
)

fig.show()

################################################################################
# **Save Plot as PNG & PDF**
################################################################################

plot_zoomed_pdf_path = os.path.join(results_dir, "dose_plot_zoomed.pdf")
fig.write_image(plot_zoomed_pdf_path, format="pdf")
print(f"Zoomed Plot saved as PDF: {os.path.abspath(plot_zoomed_pdf_path)}")