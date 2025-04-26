import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.units import inch

# Define the results directory
results_dir = "./Results/Lead/"
os.makedirs(results_dir, exist_ok=True)

# Define the list of data file paths and corresponding thickness values in cm
file_paths = [
    "./Data/Lead/100MeV1cmLead_51.txt",
    "./Data/Lead/100MeV2cmLead_51.txt",
    "./Data/Lead/100MeV3cmLead_51.txt",
    "./Data/Lead/100MeV4cmLead_51.txt",
    "./Data/Lead/100MeV5cmLead_51.txt",
    "./Data/Lead/100MeV10cmLead_51.txt",
    "./Data/Lead/100MeV15cmLead_51.txt",
    "./Data/Lead/100MeV20cmLead_51.txt",
    "./Data/Lead/100MeV30cmLead_51.txt",
    "./Data/Lead/100MeV50cmLead_51.txt"
]

thicknesses = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]  # Corresponding thickness in cm

def extract_dose_and_errors(file_path):
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

    # Round results for a scientific report
    df = df.round({
        "Thickness (cm)": 2, 
        "Dose per Proton (pSv)": 3, 
        "Normalised Error (%)": 2, 
        "Absolute Error (pSv)": 3
    })

    # Convert DataFrame to list format
    table_data = [df.columns.tolist()] + df.values.tolist()

    # Create a landscape PDF document
    pdf_table_path = os.path.join(results_dir, "dose_summary_table.pdf")
    pdf = SimpleDocTemplate(pdf_table_path, pagesize=landscape(letter))

    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#BBBBBB")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ])

    col_widths = [1.5 * inch] * len(df.columns)  
    table = Table(table_data, colWidths=col_widths)
    table.setStyle(table_style)
    pdf.build([table])

    print(f"Styled Table saved as PDF: {os.path.abspath(pdf_table_path)}")

    ################################################################################
    # Generate and Save Plot
    ################################################################################

    df["Dose per Proton (pSv)"] = df["Dose per Proton (pSv)"].replace(0, np.nan)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Thickness (cm)"],
        y=df["Dose per Proton (pSv)"],
        error_y=dict(type='data', array=df["Absolute Error (pSv)"], visible=True),
        mode='lines+markers',
        marker=dict(size=10, color='red', symbol='circle'),
        line=dict(width=3),
        name="Dose per Proton"
    ))

    fig.update_layout(
        title="Dose per Proton vs. Material Thickness (Lead)",
        title_font=dict(size=20, family="Arial"),
        xaxis=dict(
            title="Material Thickness (cm)",
            title_font=dict(size=18),
            tickfont=dict(size=14),
            showgrid=True
        ),
        yaxis=dict(
            title="Dose per Proton (pSv)",
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

    # Save Plot as PDF
    plot_pdf_path = os.path.join(results_dir, "dose_plot.pdf")
    fig.write_image(plot_pdf_path, format="pdf")
    print(f"Plot saved as PDF: {os.path.abspath(plot_pdf_path)}")
