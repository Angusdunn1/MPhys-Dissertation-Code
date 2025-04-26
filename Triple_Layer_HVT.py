import os
import re
import numpy as np

TRIPLE_LAYER_PATH = "./Data/Triple_Layers"
MC_SAMPLES        = 2000  # Monte Carlo trials for HVT error
# conversion factors: GeV/g → Sv  (1 GeV = 1.602e–10 J; assume 1 g tissue, 1 J/kg = 1 Sv)
GEV_TO_SV = 1.602e-10
# then to pSv per proton
SV_TO_PSV = 1e12


def extract_total_dose_and_error(file_path):
    """
    Parse a FLUKA ASCII output:
      - sums up the 'Data follow in a matrix' values → total dose (GeV/g)
      - reads the 'Percentage errors follow in a matrix' values → builds absolute errors
    Returns (total_dose, total_error) both in GeV/g.
    """
    dose_vals     = []
    err_perc_vals = []
    in_dose = in_err = False

    with open(file_path) as f:
        for line in f:
            if "Data follow in a matrix" in line:
                in_dose, in_err = True, False
                continue
            if in_dose and "Percentage errors follow in a matrix" in line:
                in_dose, in_err = False, True
                continue

            if in_dose:
                try:
                    dose_vals.extend(map(float, line.split()))
                except ValueError:
                    pass
            elif in_err:
                try:
                    err_perc_vals.extend(map(float, line.split()))
                except ValueError:
                    pass

    if not dose_vals:
        raise ValueError(f"No dose data in {file_path}")
    if len(err_perc_vals) != len(dose_vals):
        raise ValueError(
            f"Dose/error count mismatch in {file_path}: "
            f"{len(dose_vals)} doses vs {len(err_perc_vals)} errors"
        )

    total = np.sum(dose_vals)
    abs_errs = np.array(dose_vals) * (np.array(err_perc_vals) / 100.0)
    total_err = np.sqrt(np.sum(abs_errs**2))
    return total, total_err


def estimate_hvt_with_error(thicknesses, doses, dose_errs, n_samples=MC_SAMPLES):
    """
    1) Nominal HVT via linear interpolation on (t,d) to find t where d=D0/2.
    2) Monte Carlo: sample doses ~N(d_i, σ_i), recompute HVT many times.
    Returns (hvt_nominal, hvt_1σ_error).
    """
    # sort ascending thickness
    ts, ds, es = zip(*sorted(zip(thicknesses, doses, dose_errs)))
    ts = np.array(ts); ds = np.array(ds); es = np.array(es)
    D0 = ds[0]
    half = D0 / 2.0

    def interp_hvt(dvec):
        for i in range(1, len(dvec)):
            if dvec[i] <= half:
                x1, x2 = ts[i-1], ts[i]
                y1, y2 = dvec[i-1], dvec[i]
                if y1 == y2:
                    return np.nan
                return x1 + (half - y1)*(x2 - x1)/(y2 - y1)
        return None

    # nominal
    h0 = interp_hvt(ds)
    if h0 is None:
        return None, None

    rng = np.random.default_rng()
    h_samps = []
    for _ in range(n_samples):
        sample_ds = rng.normal(ds, es)
        sample_ds = np.clip(sample_ds, 1e-12, None)
        h = interp_hvt(sample_ds)
        if h is not None and not np.isnan(h):
            h_samps.append(h)
    if not h_samps:
        return h0, None

    return h0, np.std(h_samps, ddof=1)


def process_triple_layer(folder_path):
    """
    - Reads every *.txt (thickness in filename)
    - Builds arrays of (t, dose, dose_err) in GeV/g
    - Estimates HVT ± error (cm)
    - Computes D0 → pSv/proton and propagates its error to dose@HVT
    Returns: (hvt, hvt_err, dose_hvt_pSv, dose_hvt_err_pSv)
    """
    ths, ds, es = [], [], []
    for fname in os.listdir(folder_path):
        if not fname.endswith(".txt"):
            continue
        m = re.search(r'([\d\.]+)cm', fname)
        if not m:
            continue
        t = float(m.group(1))
        fp = os.path.join(folder_path, fname)
        try:
            total, terr = extract_total_dose_and_error(fp)
        except ValueError as e:
            print(f"Skipping {fname}: {e}")
            continue
        ths.append(t); ds.append(total); es.append(terr)

    if len(ths) < 2:
        return None, None, None, None

    hvt, hvt_err = estimate_hvt_with_error(ths, ds, es)
    if hvt is None:
        return None, None, None, None

    # D0 is at index of smallest thickness after sorting
    order = np.argsort(ths)
    D0     = ds[order[0]]
    D0_err = es[order[0]]

    # dose@HVT is D0/2, propagate error: σ(D0/2)=σ(D0)/2
    dose_hvt_raw     = D0 / 2.0
    dose_hvt_err_raw = D0_err / 2.0

    # convert to pSv/proton
    factor = GEV_TO_SV * SV_TO_PSV
    dose_hvt_pSv     = dose_hvt_raw     * factor
    dose_hvt_err_pSv = dose_hvt_err_raw * factor

    return hvt, hvt_err, dose_hvt_pSv, dose_hvt_err_pSv


def main():
    print("📊 Empirical HVT & Dose@HVT (1σ errors)\n")
    for mat in sorted(os.listdir(TRIPLE_LAYER_PATH)):
        folder = os.path.join(TRIPLE_LAYER_PATH, mat)
        if not os.path.isdir(folder):
            continue

        hvt, hvt_err, d_hvt, d_hvt_err = process_triple_layer(folder)
        if hvt is None:
            print(f"{mat:<15} Could not estimate HVT")
            continue

        # Two-decimal formatting makes the small error visible as '0.01 pSv'
        hvt_str  = f"{hvt:.2f} ± {hvt_err:.2f} cm"
        dose_str = f"{d_hvt:.2f} ± {d_hvt_err:.2f} pSv"
        print(f"{mat:<15} HVT = {hvt_str:<17} Dose@HVT = {dose_str}")


if __name__ == "__main__":
    main()
