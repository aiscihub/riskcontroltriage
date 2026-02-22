import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Helper: Otsu threshold for 1D continuous data (no labels)
# Finds a split point that maximizes between-class variance.
# ------------------------------------------------------------
def otsu_threshold(x, nbins=256):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return float(np.nan)

    # Histogram on observed range
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if np.isclose(xmin, xmax):
        return float(xmin)

    hist, bin_edges = np.histogram(x, bins=nbins, range=(xmin, xmax), density=False)
    hist = hist.astype(float)
    p = hist / hist.sum()

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    omega = np.cumsum(p)                      # class probabilities
    mu = np.cumsum(p * bin_centers)           # class means (unnormalized)
    mu_t = mu[-1]

    # between-class variance: (mu_t*omega - mu)^2 / (omega*(1-omega))
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    k = int(np.nanargmax(sigma_b2))
    return float(bin_centers[k])

# ------------------------------------------------------------
# Calibration protocol per ligand:
# - Contact cutoff: Otsu split on f_contact_20ns, then clamp to [0.05, 0.95]
# - RMSD cutoff:    Otsu split on rmsd_20ns (finite), then clamp to [0.3Å, 5.0Å]
# - Drift cutoff:   fixed physical escape threshold (default 6Å)
#
# Optional "safety_bias":
#   +0.0 means purely data-driven split
#   >0 shifts thresholds slightly more "risk-averse" (fewer unstable labels),
#   <0 shifts thresholds more "aggressive" (more unstable labels).
# ------------------------------------------------------------
def calibrate_cutoffs_from_20ns(
        df_20ns: pd.DataFrame,
        ligand_col="ligand_name",
        f_contact_col="f_contact_20ns",
        rmsd_col="rmsd_20ns",
        drift_col="ligand_drift",
        drift_cutoff=6.0,
        nbins=256,
        safety_bias=0.0,
):
    """
    Returns dict: ligand_name -> dict(F_CONTACT_CUTOFF, RMSD_CUTOFF, DRIFT_CUTOFF, diagnostics)
    Uses only endpoint (20 ns) columns in df_20ns.
    """

    out = {}

    for ligand, sub in df_20ns.groupby(ligand_col):
        f_contact = sub[f_contact_col].to_numpy(dtype=float)
        rmsd = sub[rmsd_col].to_numpy(dtype=float)

        # 1) Data-driven split thresholds
        f_thr = otsu_threshold(f_contact, nbins=nbins)
        r_thr = otsu_threshold(rmsd[np.isfinite(rmsd)], nbins=nbins)

        # 2) Clamp to reasonable physical ranges (prevents weird thresholds on tiny/noisy sets)
        if np.isfinite(f_thr):
            f_thr = float(np.clip(f_thr, 0.05, 0.95))
        if np.isfinite(r_thr):
            r_thr = float(np.clip(r_thr, 0.3, 5.0))

        # 3) Optional bias adjustment (protocol knob, still uses only endpoint data)
        #    - For contact: lower cutoff => more unstable. So +bias raises cutoff slightly.
        #    - For RMSD: higher cutoff => fewer unstable. So +bias raises cutoff slightly.
        if np.isfinite(f_thr):
            f_thr = float(np.clip(f_thr + 0.05 * safety_bias, 0.05, 0.95))
        if np.isfinite(r_thr):
            r_thr = float(np.clip(r_thr + 0.25 * safety_bias, 0.3, 5.0))

        # Diagnostics: how the endpoint distribution looks
        diag = {
            "n": int(len(sub)),
            "f_contact_median": float(np.nanmedian(f_contact)) if len(f_contact) else np.nan,
            "f_contact_q25": float(np.nanpercentile(f_contact, 25)) if len(f_contact) else np.nan,
            "f_contact_q75": float(np.nanpercentile(f_contact, 75)) if len(f_contact) else np.nan,
            "rmsd_median": float(np.nanmedian(rmsd)) if len(rmsd) else np.nan,
            "rmsd_q25": float(np.nanpercentile(rmsd[np.isfinite(rmsd)], 25)) if np.isfinite(rmsd).any() else np.nan,
            "rmsd_q75": float(np.nanpercentile(rmsd[np.isfinite(rmsd)], 75)) if np.isfinite(rmsd).any() else np.nan,
            "otsu_contact": f_thr,
            "otsu_rmsd": r_thr,
        }

        out[ligand] = dict(
            F_CONTACT_CUTOFF=f_thr,
            RMSD_CUTOFF=r_thr,
            DRIFT_CUTOFF=float(drift_cutoff),
            diagnostics=diag,
        )

    return out

# ------------------------------------------------------------
# Apply rule (your existing labeling logic), but parameterized.
# ------------------------------------------------------------
def label_unstable_from_endpoints(
        f_contact: float,
        rmsd_late: float,
        drift: float,
        F_CONTACT_CUTOFF: float,
        RMSD_CUTOFF: float,
        DRIFT_CUTOFF: float,
):
    return int(
        (f_contact <= F_CONTACT_CUTOFF)
        or (not np.isnan(rmsd_late) and rmsd_late >= RMSD_CUTOFF)
        or (np.isnan(rmsd_late) and drift is not None and drift >= DRIFT_CUTOFF)
    )

# ------------------------------------------------------------
# Example usage:
# ------------------------------------------------------------
if __name__ == "__main__":
    # Load your endpoint summary (one row per pocket/replica with 20ns endpoints)
    df = pd.read_csv("fingerprint_summary_with_components_even_ac_4d_drift_20260214_flipped4.csv")

    # If your file contains multiple time_ns rows, filter to the 20ns endpoint rows
    # (adjust depending on your schema)
    if "time_ns" in df.columns:
        df_20 = df[df["time_ns"] == 20.0].copy()
    else:
        df_20 = df.copy()

    cutoffs = calibrate_cutoffs_from_20ns(df_20, safety_bias=0.0)

    for ligand, theta in cutoffs.items():
        print(ligand, theta["F_CONTACT_CUTOFF"], theta["RMSD_CUTOFF"], theta["DRIFT_CUTOFF"])