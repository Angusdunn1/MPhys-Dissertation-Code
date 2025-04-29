import os
import re
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def exp_decay(x, D0, mu):
    """Simple exponential: D(x) = D0·exp(–μ·x)."""
    return D0 * np.exp(-mu * x)

def fixed_decay(x, mu):
    """For non-Al materials we anchor D0 at 36.731841 pSv and fit only mu."""
    D0 = 36.731841
    return D0 * np.exp(-mu * x)

def get_material_data(base_path="./Data"):
    materials = {}
    for material_name in os.listdir(base_path):
        mat_dir = os.path.join(base_path, material_name)
        if not os.path.isdir(mat_dir):
            continue

        # Special handling for triple layers
        if material_name == "Triple_Layers":
            for combo in os.listdir(mat_dir):
                combo_dir = os.path.join(mat_dir, combo)
                if not os.path.isdir(combo_dir):
                    continue

                fpaths, thicks = [], []
                for fname in os.listdir(combo_dir):
                    if not fname.endswith(".txt"):
                        continue
                    m = re.search(r'([\d\.]+)cm', fname)
                    if not m:
                        continue
                    th = float(m.group(1))
                    fpaths.append(os.path.join(combo_dir, fname))
                    thicks.append(th)

                if fpaths:
                    sorted_pairs = sorted(zip(thicks, fpaths))
                    ts, fps = zip(*sorted_pairs)
                    materials[combo] = {
                        "thicknesses": list(ts),
                        "file_paths": list(fps)
                    }

        else:
            # Normal material folder
            fpaths, thicks = [], []
            for fname in os.listdir(mat_dir):
                if not fname.endswith(".txt"):
                    continue
                m = re.search(r'([\d\.]+)cm', fname)
                if not m:
                    continue
                th = float(m.group(1))
                fpaths.append(os.path.join(mat_dir, fname))
                thicks.append(th)

            if fpaths:
                sorted_pairs = sorted(zip(thicks, fpaths))
                ts, fps = zip(*sorted_pairs)
                materials[material_name] = {
                    "thicknesses": list(ts),
                    "file_paths": list(fps)
                }

    return materials

def extract_dose_and_errors(file_path):
    dose_vals, err_vals = [], []
    in_dose = in_err = False
    for line in open(file_path):
        if "Data follow in a matrix" in line:
            in_dose, in_err = True, False
            continue
        if "Percentage errors follow in a matrix" in line:
            in_dose, in_err = False, True
            continue

        if in_dose:
            try:
                dose_vals.extend([float(x) for x in line.split()])
            except ValueError:
                pass
        elif in_err:
            try:
                err_vals.extend([float(x) for x in line.split()])
            except ValueError:
                pass

    return np.array(dose_vals), np.array(err_vals)

absorption_data = []
failed_materials = []

def process_material(material_name, file_paths, thicknesses, plot_dir="./Plots"):
    results = []
    for fp, thick in zip(file_paths, thicknesses):
        doses, errs = extract_dose_and_errors(fp)
        if doses.size == 0 or errs.size == 0:
            continue

        # sum dose, convert to pSv per proton
        total_dose_GeV_per_g = doses.sum()
        total_dose_Sv      = total_dose_GeV_per_g * 1.602e-10
        dose_pSv           = total_dose_Sv * 1e12

        mean_pct_err = errs.mean()
        abs_err      = dose_pSv * (mean_pct_err / 100.0)

        results.append([thick, dose_pSv, abs_err])

    if len(results) < 3:
        print(f"Not enough data for {material_name}, skipping.")
        failed_materials.append(material_name)
        return

    df = pd.DataFrame(results, columns=["Thickness", "Dose", "AbsError"]).dropna()
    x, y, yerr = df["Thickness"].values, df["Dose"].values, df["AbsError"].values

    os.makedirs(plot_dir, exist_ok=True)

    try:
        if material_name.lower() == "aluminium":
            # --- LOG-LINEAR FIT ON 0–8 cm FOR ALUMINIUM ---
            mask = x <= 8.0
            x_fit, y_fit, yerr_fit = x[mask], y[mask], yerr[mask]

            # convert to log-space
            log_y    = np.log(y_fit)
            log_yerr = yerr_fit / y_fit

            def log_model(x, lnD0, mu):
                return lnD0 - mu * x

            popt, pcov = curve_fit(
                log_model, x_fit, log_y,
                sigma=log_yerr, absolute_sigma=True,
                p0=(np.log(y_fit[0]), 0.2),
                bounds=([-np.inf, 1e-4], [np.inf, np.inf]),
                maxfev=10000
            )

            lnD0_fit, mu_fit = popt
            D0_fit           = np.exp(lnD0_fit)
            lnD0_err         = np.sqrt(pcov[0,0])
            D0_err           = D0_fit * lnD0_err
            mu_err           = np.sqrt(pcov[1,1])

            # half-value thickness and its error
            hvt     = np.log(2) / mu_fit
            hvt_err = (np.log(2) / mu_fit**2) * mu_err

            # dose per proton at the HVT and its uncertainty
            dose_hvt     = D0_fit / 2
            dose_hvt_err = D0_err  / 2

            absorption_data.append({
                "Material":                  material_name,
                "μ (cm⁻¹)":                  mu_fit,
                "μ Error":                   mu_err,
                "Half-Value Thickness (cm)": hvt,
                "HVT Error":                 hvt_err,
                "Dose at HVT (pSv)":         dose_hvt,
                "Dose at HVT Error":         dose_hvt_err
            })

            # plotting
            plt.figure()
            plt.errorbar(x, y, yerr=yerr, fmt='o', color='red', label='Data')
            x_line = np.linspace(0, x_fit.max(), 200)
            plt.plot(x_line, exp_decay(x_line, D0_fit, mu_fit),
                     color='black', label=f'Fit: μ={mu_fit:.4f}')
            plt.axhline(D0_fit / 2, linestyle='--', color='blue', label='Half Dose')
            plt.yscale('log')
            plt.xlabel("Aluminium Thickness (cm)")
            plt.ylabel("Dose per Proton (pSv)")
            plt.title("Aluminium – Log-Linear Fit (0–8 cm)")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{material_name}_fit.png"))
            plt.close()

        else:
            # --- FIXED-D0 FIT FOR OTHER MATERIALS ---
            popt, pcov = curve_fit(
                fixed_decay, x, y,
                sigma=yerr, absolute_sigma=True,
                p0=[0.1], bounds=([0],[np.inf])
            )
            mu_fit = popt[0]
            mu_err = np.sqrt(pcov[0,0])

            # half-value thickness and its error
            hvt     = np.log(2) / mu_fit
            hvt_err = (np.log(2) / mu_fit**2) * mu_err

            # dose at HVT for fixed D0 (no D0 uncertainty)
            D0_fixed     = 36.731841
            dose_hvt     = D0_fixed / 2
            dose_hvt_err = 0.0

            absorption_data.append({
                "Material":                  material_name,
                "μ (cm⁻¹)":                  mu_fit,
                "μ Error":                   mu_err,
                "Half-Value Thickness (cm)": hvt,
                "HVT Error":                 hvt_err,
                "Dose at HVT (pSv)":         dose_hvt,
                "Dose at HVT Error":         dose_hvt_err
            })

            # plotting
            plt.figure()
            plt.errorbar(x, y, yerr=yerr, fmt='o', color='red', label='Data')
            x_line = np.linspace(0, x.max(), 200)
            plt.plot(x_line, fixed_decay(x_line, mu_fit),
                     color='black', label=f'Fit: μ={mu_fit:.4f}')
            plt.axhline(D0_fixed / 2, linestyle='--', color='blue', label='Half Dose')
            plt.yscale('log')
            plt.xlabel("Additional Shield Thickness (cm)")
            plt.ylabel("Dose per Proton (pSv)")
            plt.title(f"{material_name} – Additional Shielding Fit")
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f"{material_name}_fit.png"))
            plt.close()

    except Exception as e:
        print(f"Fit failed for {material_name}: {e}")
        failed_materials.append(material_name)


if __name__ == "__main__":
    materials = get_material_data("./Data")
    for mat, data in materials.items():
        process_material(mat, data["file_paths"], data["thicknesses"])

    # write out summary
    df_out = pd.DataFrame(absorption_data).round(5)
    df_out.to_csv("absorption_coefficients_summary.csv", index=False)

    print("\nProcessing complete.")
    if failed_materials:
        print("Fit failed for:")
        for m in failed_materials:
            print(" -", m)
