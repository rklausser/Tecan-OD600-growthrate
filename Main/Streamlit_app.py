# ---------------------------------------------------------
# Streamlit app for the full TECAN OD600 pipeline:
#   Dataimport.py  ->  Dataprocessing.py  ->  Plot_results.py
#
# Run with:  python -m streamlit run Main/Streamlit_app.py
#
# Upload:
#   * up to 10 TECAN result .xlsx files (see Test_Dataset for format)
#   * one well-assignment sheet
# Toggle "Download template" to grab example files for both.
# ---------------------------------------------------------

import io
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend so plt.figure works inside Streamlit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Make the pipeline modules importable from the same folder
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import Dataimport as dim  # noqa: E402  (import after sys.path setup)
import Dataprocessing as dproc  # noqa: E402
import Plot_results as pr  # noqa: E402


# ---------------------------------------------------------
# Template generators (in-memory)
# ---------------------------------------------------------

ASSIGNMENT_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# 24-well plate used in this project (GRE24ft): A1 -> D6
PLATE_WELLS = [f"{row}{col}" for row in "ABCD" for col in range(1, 7)]


def tecan_template_bytes():
    """
    Build a minimal but valid TECAN 'Result sheet' that read_tecan()
    understands (parameter in column A, value in column E, plus a
    'Cycle Nr.' header row and a few synthetic cycles).
    """
    header = ["Cycle Nr.", "Time [s]", "Temp. [C]"] + PLATE_WELLS
    meta = [
        ("Method name", "RKRKL_Example"),
        ("Application", "SparkControl"),
        ("Date", "2026-01-01"),
        ("Time", "12:00"),
        ("Start Time", "2026-01-01 12:00:00"),
        ("Plate", "GRE24ft"),
        ("Plate area", "A1-D6"),
        ("Plate layout", "GRE24ft"),
        ("Kinetic duration [hhmmss]", "01:00:00"),
        ("Interval time [hhmmss]", "00:10:00"),
        ("Measurement wavelength [nm]", "600"),
        ("Number of flashes", "10"),
    ]
    ncycles = 12
    ncols = len(header)
    header_row = len(meta) + 1              # one blank row before the header
    nrows = header_row + 1 + ncycles

    raw = pd.DataFrame(np.nan, index=range(nrows), columns=range(ncols), dtype=object)
    for i, (param, value) in enumerate(meta):
        raw.iloc[i, 0] = param
        raw.iloc[i, 4] = value

    raw.iloc[header_row, :] = header
    for k in range(ncycles):
        r = header_row + 1 + k
        raw.iloc[r, 0] = k + 1
        raw.iloc[r, 1] = k * 600.0
        raw.iloc[r, 2] = 30.0
        for j, well in enumerate(PLATE_WELLS):
            raw.iloc[r, 3 + j] = 0.12 + 0.0015 * k + 0.005 * np.random.rand()

    buf = io.BytesIO()
    raw.to_excel(buf, sheet_name="Result sheet", index=False, header=False, engine="openpyxl")
    return buf.getvalue()


def assignment_template_bytes():
    """
    Build a well-assignment template with the required
    well / condition / replicate columns (extra parameter columns
    can simply be added by the user).

    Conditions are grouped in blocks of three replicates so the sheet
    already passes the duplicate-replicate validation (8 conditions x 3).
    """
    n = len(PLATE_WELLS)
    template = pd.DataFrame(
        {
            "well": PLATE_WELLS,
            "condition": [f"Condition_{(i // 3) + 1}" for i in range(n)],
            "replicate": ([1, 2, 3] * (n // 3))[:n],
        }
    )
    buf = io.BytesIO()
    template.to_excel(buf, sheet_name="Well assignment", index=False, engine="openpyxl")
    return buf.getvalue()


# ---------------------------------------------------------
# Processing (cached so the uploads are only parsed once)
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_pipeline(tecan_bytes: tuple, assignment_bytes: bytes):
    """
    Write the uploaded TECAN files to a temp folder, then run the
    Dataimport -> Dataprocessing pipeline. Returns the summary DataFrame.
    """
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, blob in enumerate(tecan_bytes):
            p = Path(tmp) / f"tecan_upload_{i}.xlsx"
            p.write_bytes(blob)
            paths.append(p)

        data_df, metadata_df = dim.concatenate_tecan_files(
            paths, start_time_parameter="Start Time"
        )
        assignment_df = pd.read_excel(io.BytesIO(assignment_bytes))

    summary_df = dproc.process_tecans(data_df, metadata_df, assignment_df)
    return summary_df


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(page_title="TECAN OD600 pipeline", layout="wide")
st.title("TECAN OD600 growth-rate pipeline")

# ---------------------------------------------------------
# Left panel - Step 1: uploads (import fields)
# ---------------------------------------------------------

with st.sidebar:
    st.header("1. Import data")

    st.subheader("TECAN data files")
    tecan_files = st.file_uploader(
        "TECAN result sheets (.xlsx) - up to 10 files",
        type=["xlsx"],
        accept_multiple_files=True,
    )
    if tecan_files and len(tecan_files) > 10:
        st.warning("Only the first 10 uploaded files will be processed.")
    st.download_button(
        "Download TECAN template",
        data=tecan_template_bytes(),
        file_name="tecan_template.xlsx",
        mime=XLSX_MIME,
    )

    st.subheader("Well assignment sheet")
    assignment_file = st.file_uploader(
        "Well assignment sheet (.xlsx)",
        type=["xlsx"],
    )
    st.download_button(
        "Download assignment template",
        data=assignment_template_bytes(),
        file_name="assignment_template.xlsx",
        mime=ASSIGNMENT_MIME,
    )

# ---------------------------------------------------------
# Step 2 - run the pipeline
# ---------------------------------------------------------

df = None
process_error = None

if not tecan_files or not assignment_file:
    st.info("Please upload TECAN data files and a well assignment sheet to run the pipeline.")
elif len(tecan_files) > 10:
    process_error = "Please upload at most 10 TECAN files."
else:
    try:
        df = run_pipeline(
            tuple(f.getvalue() for f in tecan_files),
            assignment_file.getvalue(),
        )
    except Exception as exc:  # noqa: BLE001 - report any pipeline failure to the user
        process_error = f"{type(exc).__name__}: {exc}"

if process_error:
    st.error(f"Processing failed: {process_error}")

if df is not None:
    st.session_state["summary_df"] = df
    n_conditions = df["condition"].nunique() if "condition" in df.columns else "?"
    st.success(
        f"Processed {len(tecan_files)} file(s) into "
        f"{df.shape[0]:,} summary rows across {n_conditions} conditions."
    )

summary_df = st.session_state.get("summary_df")


# ---------------------------------------------------------
# Step 3 - plot configuration (sidebar) + figure
# ---------------------------------------------------------

with st.sidebar:
    st.header("2. Plot configuration")

    if summary_df is None:
        st.info("Upload data and a well-assignment sheet, then run the pipeline.")

    else:
        params = pr.parameter_columns(summary_df)

        metric = st.radio("Y metric", ["OD600", "mu (growth rate)"])
        if metric == "OD600":
            y_mean, y_se, ylabel = "OD_mean", "OD_se", "OD600"
        else:
            y_mean, y_se, ylabel = "mu_mean", "mu_se", "µ [h⁻¹]"

        # Seed sensible defaults (mirrors pr.default_config but index-safe,
        # so a minimal assignment sheet with few parameters still works).
        def _safe_default(df, y_mean, y_se, ylabel):
            cfg = pr.PlotConfig(y_mean=y_mean, y_se=y_se, ylabel=ylabel)
            params_ = pr.parameter_columns(df)
            if len(params_) >= 2:
                cfg.color_var = params_[1]
            if len(params_) >= 5:
                cfg.style_var = params_[4]
            return cfg

        default = _safe_default(summary_df, y_mean=y_mean, y_se=y_se, ylabel=ylabel)

        default_color = default.color_var if default.color_var in params else (params[0] if params else None)
        default_style = default.style_var if default.style_var in params else (params[1] if len(params) > 1 else (params[0] if params else None))

        color_var = st.selectbox(
            "Color by", params, index=params.index(default_color) if default_color in params else 0
        )
        style_var = st.selectbox(
            "Line style by", params,
            index=params.index(default_style) if default_style in params else 0,
        )

        facet_vars = st.multiselect(
            "Facet by (one panel per combination)",
            [p for p in params if p not in (color_var, style_var)],
            default=[],
        )

        ncol = st.number_input("Max facet columns", min_value=1, value=3, step=1)
        share_y = st.checkbox("Share y-scale across all facets", value=True)

        with st.expander("Filters (subset the data)"):
            filters = {}
            for p in params:
                values = pr.unique(summary_df, p)
                selected = st.multiselect(p, values, default=values)
                if selected and len(selected) < len(values):
                    filters[p] = selected

        with st.expander("Colors & line styles"):
            cvals = pr.unique(summary_df, color_var)
            color_order = []
            for i, v in enumerate(cvals):
                color_order.append(
                    st.color_picker(
                        f"color {v}", value=pr.COLOR_BASE[i % len(pr.COLOR_BASE)]
                    )
                )

            svals = pr.unique(summary_df, style_var)
            style_order = []
            for v in svals:
                style_order.append(st.selectbox(f"linestyle {v}", pr.LINESTYLES, index=0))

        cfg = pr.PlotConfig(
            color_var=color_var,
            color_order=color_order,
            style_var=style_var,
            style_order=style_order,
            facet_vars=facet_vars,
            y_mean=y_mean,
            y_se=y_se,
            ylabel=ylabel,
            ncol=int(ncol),
            filters=filters,
            share_y=share_y,
        )

        st.session_state["plot_cfg"] = cfg


if summary_df is not None:
    cfg = st.session_state.get("plot_cfg")
    if cfg is None:
        cfg = pr.default_config(summary_df)

    st.header("Results")
    st.subheader(f"{cfg.ylabel} over time")
    st.caption(
        f"color: {cfg.color_var} | line: {cfg.style_var} | "
        f"facets: {cfg.facet_vars or 'none'} | filters: {cfg.filters or 'none'}"
    )

    fig = pr.plot_mean_metric(summary_df, cfg)
    fig.tight_layout()
    st.pyplot(fig)
