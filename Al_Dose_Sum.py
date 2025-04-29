import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.units import inch

# Define the results directory
results_dir = "./Results/Aluminium/"

# Ensure the directory exists
os.makedirs(results_dir, exist_ok=True)

# Define the list of data file paths and corresponding thickness values in cm
file_paths = [
    "./Data/Aluminium/100MeVnoShield_51.txt",
    "./Data/Aluminium/100MeV1cm_51.txt",
    "./Data/Aluminium/100MeV2cm_51.txt",
    "./Data/Aluminium/100MeV2.25cm_51.txt",
    "./Data/Aluminium/100MeV2.5cm_51.txt",
    "./Data/Aluminium/100MeV2.75cm_51.txt",
    "./Data/Aluminium/100MeV3cm_51.txt",
    "./Data/Aluminium/100MeV3.25cm_51.txt",
    "./Data/Aluminium/100MeV3.5cm_51.txt",
    "./Data/Aluminium/100MeV3.75cm_51.txt",
    "./Data/Aluminium/100MeV4cm_51.txt",
    "./Data/Aluminium/100MeV4.5_51.txt",
    "./Data/Aluminium/100MeV6cm_51.txt",
    "./Data/Aluminium/100MeV8cm_51.txt",
    "./Data/Aluminium/100MeV10cm_51.txt",
    "./Data/Aluminium/100MeV20cm_51.txt",
]

thicknesses = [0, 1, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.5, 6, 8, 10, 20]  # Corresponding thickness in cm

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

if __name__ == "__main__":

    # Arrays to store results for each file
    results = []

    for file_path, thickness in zip(file_paths, thicknesses):
        dose_values, error_values = extract_dose_and_errors(file_path)

        # Step 1: Sum the GeV/g/proton values
        total_dose_gev_per_g = np.sum(dose_values)

        # Step 2: Convert GeV/g -> Sv
        total_dose_Sv = total_dose_gev_per_g * 1.602e-10

        # Step 3: Convert Sv -> pSv
        dose_per_proton_pSv = total_dose_Sv * 1e12

        # Step 4: Error calculation
        total_error = np.sum(error_values)
        normalised_error = total_error / 1000  # assuming 1000 bins
        error_in_dose = dose_per_proton_pSv * (normalised_error / 100.0)

        results.append([thickness, dose_per_proton_pSv, normalised_error, error_in_dose])

    # Convert results into a DataFrame
    df = pd.DataFrame(results, columns=["Thickness (cm)", "Dose per Proton (pSv)", "Normalised Error (%)", "Absolute Error (pSv)"])
    
    # Print table in terminal
    print(df.to_string(index=False))


    ################################################################################
    # Convert Table to a PDF Using ReportLab
    ################################################################################

# Define PDF table path
pdf_table_path = os.path.join(results_dir, "dose_summary_table.pdf")

# **Round results for a scientific report** (2-3 decimal places where appropriate)
df = df.round({
    "Thickness (cm)": 2, 
    "Dose per Proton (pSv)": 3, 
    "Normalised Error (%)": 2, 
    "Absolute Error (pSv)": 3
})

# Convert DataFrame to list format
table_data = [df.columns.tolist()] + df.values.tolist()

# Create a **landscape PDF document** for readability
pdf = SimpleDocTemplate(pdf_table_path, pagesize=landscape(letter))

# Define a **lighter grey header** for a clean, professional look
table_style = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#BBBBBB")),  # **Lighter grey header**
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Black text in header
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Center align all text
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Bold font for header
    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),  # Regular font for body
    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),  # Padding for header row
    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),  # Light grey for alternating rows
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),  # Alternating row colors
    ('GRID', (0, 0), (-1, -1), 0.5, colors.black),  # Fine black grid lines
])

# Set **column widths** to improve spacing
col_widths = [1.5 * inch] * len(df.columns)  

# Create the table
table = Table(table_data, colWidths=col_widths)
table.setStyle(table_style)

# Build the PDF document
pdf.build([table])

print(f"Styled Table saved as PDF: {os.path.abspath(pdf_table_path)}")

    ################################################################################
    # Generate and Save Plot
    ################################################################################

df["Dose per Proton (pSv)"] = df["Dose per Proton (pSv)"].replace(0, np.nan)

fig = go.Figure()

# First trace: all points except 3.5 cm (red points, red error bars)
mask_red = df["Thickness (cm)"] != 3.5
fig.add_trace(go.Scatter(
    x=df["Thickness (cm)"][mask_red],
    y=df["Dose per Proton (pSv)"][mask_red],
    error_y=dict(
        type='data',
        array=df["Absolute Error (pSv)"][mask_red],
        visible=True,
        color='red'  # Red error bars
    ),
    mode='lines+markers',
    marker=dict(
        size=10,
        color='red',
        symbol='circle'
    ),
    line=dict(
        width=3,
        color='red'
    ),
    name="Dose per Proton (other points)"
))

# Second trace: only 3.5 cm point (blue point, blue error bar)
mask_blue = df["Thickness (cm)"] == 3.5
fig.add_trace(go.Scatter(
    x=df["Thickness (cm)"][mask_blue],
    y=df["Dose per Proton (pSv)"][mask_blue],
    error_y=dict(
        type='data',
        array=df["Absolute Error (pSv)"][mask_blue],
        visible=True,
        color='blue'  # Blue error bar
    ),
    mode='markers',  # No line, just marker
    marker=dict(
        size=12,  # (Optional) slightly bigger blue point
        color='blue',
        symbol='circle'
    ),
    name="Dose per Proton (3.5 cm)"
))

fig.update_layout(
    title="Dose per Proton vs. Material Thickness (Aluminium)",
    title_font=dict(size=30, family="Arial"),
    xaxis=dict(
        title="Material Thickness (cm)",
        title_font=dict(size=26),
        tickfont=dict(size=22),
        showgrid=True
    ),
    yaxis=dict(
        title="Dose per Proton (pSv)",
        title_font=dict(size=26),
        tickfont=dict(size=22),
        type="log",
        showgrid=True
    ),
    template="simple_white",
    hovermode="x",
    font=dict(family="Arial", size=22),
    margin=dict(l=100, r=100, t=80, b=80),
    showlegend=False  # <<-- This hides the key
)

fig.show()

    ################################################################################
    # Save Plot as PDF
    ################################################################################

plot_pdf_path = os.path.join(results_dir, "dose_plot.pdf")
fig.write_image(plot_pdf_path, format="pdf", width=1200, height=900)
print(f"Plot saved as PDF: {os.path.abspath(plot_pdf_path)}")