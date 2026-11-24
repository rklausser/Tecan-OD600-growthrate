import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from pathlib import Path


# --------------------------
# constants and helpers
# --------------------------

# Standard wells on the GRE24ft (24 well) plate used here: A1->D6
WELL_ROWS = "ABCD"
WELL_COLUMNS = [f"{row}{col}" for row in WELL_ROWS for col in range(1, 7)]


def find_time_column(data_df):
    """Return the single column that holds time, raising if ambiguous."""
    time_columns = [c for c in data_df.columns if "Time" in str(c)]

    if len(time_columns) == 0:
        raise ValueError("No time columns found")

    if len(time_columns) > 1:
        raise ValueError(
            f"Multiple time columns found: {time_columns}; only 1 time column allowed"
        )

    return time_columns[0]


def check_well_columns(data_df, well_columns=WELL_COLUMNS):
    """Raise if any expected well column is missing from the data."""
    missing_wells = [w for w in well_columns if w not in data_df.columns]

    if missing_wells:
        raise ValueError(f"Missing well columns: {missing_wells}")

    return well_columns


def read_well_assignments(assignment_df):
    """
    Convert a well assignment DataFrame into a dict of well -> {column: value}.

    Parameters
    ----------
    assignment_df : pandas.DataFrame
        Must contain at least the columns 'well', 'condition' and 'replicate'.
        Any additional columns are treated as extra experimental parameters.

    Returns
    -------
    well_assignments : dict
        Mapping of well name to a dict of its parameters (replicate + extras).
    """
    required = ["well", "condition", "replicate"]
    missing = [c for c in required if c not in assignment_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    meta = assignment_df.copy()
    #make sure that replicates are integers
    meta["replicate"] = meta["replicate"].astype(int)

    well_assignments = {
        row["well"]: {
            col: row[col]
            for col in meta.columns
            if col != "well"
        }
        for _, row in meta.iterrows()
    }

    return well_assignments


# ---------------------------------------------------------
# well assignment validation
# ---------------------------------------------------------

def validate_assignments(data_df, well_assignments):
    """
    Check that well assignments are complete and consistent.

    Parameters
    ----------
    data_df : pandas.DataFrame
        DataFrame returned by read_tecan().

    well_assignments : dict
        Dictionary mapping well names to condition and replicate.

    Returns
    -------
    None
        Raises ValueError if an assignment is invalid.
    """

    # -----------------------------------------------------
    # 1. Check that all assigned wells exist in the data
    # -----------------------------------------------------

    missing_wells = [
        well
        for well in well_assignments
        if well not in data_df.columns
    ]

    if missing_wells:
        raise ValueError(
            f"Assigned wells not found in data: {missing_wells}"
        )

    # -----------------------------------------------------
    # 2. Check that every assignment has the required fields
    # -----------------------------------------------------

    for well, assignment in well_assignments.items():

        if "condition" not in assignment:
            raise ValueError(
                f"well {well} has no condition assigned."
            )

        if "replicate" not in assignment:
            raise ValueError(
                f"well {well} has no replicate number."
            )

    # -----------------------------------------------------
    # 3. Check that replicate numbers are unique
    #    within each condition
    # -----------------------------------------------------

    condition_replicates = {}

    for well, assignment in well_assignments.items():

        condition = assignment["condition"]
        replicate = assignment["replicate"]

        if condition not in condition_replicates:
            condition_replicates[condition] = []

        condition_replicates[condition].append(replicate)

    for condition, replicates in condition_replicates.items():

        if len(replicates) != len(set(replicates)):
            raise ValueError(
                f"Duplicate replicate numbers in condition "
                f"'{condition}': {replicates}"
            )

    print("well assignments validated successfully.")


# ---------------------------------------------------------
# Convert data to long form DataFrame
# ---------------------------------------------------------

def assign_conditions(data_df, well_assignments):
    """
    Convert TECAN well data into long format and assign
    conditions and replicate numbers.
    """

    validate_assignments(data_df, well_assignments)

    well_columns = list(well_assignments.keys())

    # Find the time column
    time_column = find_time_column(data_df)

    # Convert wide → long
    long_df = data_df.melt(
        id_vars=["Cycle Nr.", time_column],
        value_vars=well_columns,
        var_name="well",
        value_name="Y Data"
    )

    # Create assignment DataFrame
    assignment_df = pd.DataFrame.from_dict(
        well_assignments,
        orient="index"
    )

    assignment_df.index.name = "well"

    assignment_df = assignment_df.reset_index()

    # Merge assignments onto the measurement data
    long_df = long_df.merge(
        assignment_df,
        on="well",
        how="left"
    )

    return long_df


# ---------------------------------------------------------
# Filter for spikes in measurements
# ---------------------------------------------------------

def remove_spikes(long_df, window=5, threshold=2):
    """
    Remove isolated spikes using a rolling median and MAD.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Long-format TECAN data.

    window : int
        Number of points in the rolling window.

    threshold : float
        Spike threshold in robust standard deviations.
        if threshold * robust standard dev > absolute deviation = spike

    Returns
    -------
    filtered_df : pandas.DataFrame
        Original data plus filtered values and spike flags.
    """

    filtered_df = long_df.copy()

    # -----------------------------------------------------
    # 1. Rolling median
    # -----------------------------------------------------

    filtered_df["rolling_median"] = (
        filtered_df
        .groupby("well")["Y Data"]
        .transform(
            lambda x: x.rolling( #Note: lambda are "anonymous function" to be used once and never called again
                window=window,
                center=True,
                min_periods=1
            ).median()
        )
    )

    # -----------------------------------------------------
    # 2. Absolute deviation from rolling median
    # -----------------------------------------------------

    filtered_df["absolute_deviation"] = (
        filtered_df["Y Data"]
        - filtered_df["rolling_median"]
    ).abs()

    # -----------------------------------------------------
    # 3. Rolling MAD
    # -----------------------------------------------------

    filtered_df["rolling_mad"] = (
        filtered_df
        .groupby("well")["absolute_deviation"]
        .transform(
            lambda x: x.rolling(
                window=window,
                center=True,
                min_periods=1
            ).median()
        )
    )

    # -----------------------------------------------------
    # 4. Convert MAD to robust standard deviation
    # -----------------------------------------------------

    filtered_df["robust_std"] = (
        1.4826 * filtered_df["rolling_mad"]
    )

    # -----------------------------------------------------
    # 5. Identify spikes
    # -----------------------------------------------------

    filtered_df["is_spike"] = (
        filtered_df["absolute_deviation"]
        > threshold * filtered_df["robust_std"]
    )

    # -----------------------------------------------------
    # 6. Replace spikes
    # -----------------------------------------------------

    filtered_df["Y Data filtered"] = (
        filtered_df["Y Data"]
        .where(
            ~filtered_df["is_spike"],
            filtered_df["rolling_median"]
        )
    )

    return filtered_df


# --------------------------
# Define Smoothing function (Savitzki-Golay)
# --------------------------

def smooth_with_savgol(group, window_length=11, polyorder=2):
    group = group.sort_values("Time [s]").copy()

    y = group["Y Data filtered"].to_numpy()

    group["Y Data smoothed"] = savgol_filter(
        y,
        window_length=window_length, #Must be odd for SG to work
        polyorder=polyorder # higher order improves fit but may fit noise
    )

    return group


def calculate_growthrate(group):
    """
    calculates growth rate from smoothed data.

    :returns: dataframe with parameters and growth rate in h^-1
    """

    group = group.sort_values("Time [s]").copy()

    t = group["Time [s]"].to_numpy() / 3600 # convert time to h
    Y = group["Y Data smoothed"].to_numpy()

    mu = np.full(Y.shape, np.nan, dtype=float)

    dt = np.diff(t)

    valid = ((
        Y[1:] > 0)
        & (Y[:-1] > 0)
        & (dt > 0)
    )

    mu[1:] = np.where(
        valid,
        (np.log(Y[1:]) - np.log(Y[:-1])) / dt,
        np.nan
    )

    group["mu"] = mu

    return group


# ---------------------------------------------------------
# calculate summary of OD ( mean and se) before plotting
# ---------------------------------------------------------

def calculate_summary(mu_df, OD_column, mu_column, well_assignments):
    """
    Calculate mean, standard deviation, number of replicates,
    and standard error for each condition and time point.
    """
    parameter_columns = sorted({
        key
        for assignment in well_assignments.values()
        for key in assignment
        if key != "replicate"
    })

    #only keep columns that are present in mu_df as a safety check
    parameter_columns = [c for c in parameter_columns if c in mu_df.columns]

    summary_df = (
        mu_df
        .groupby(["Time [s]"] + parameter_columns)
        .agg(
            OD_mean=(OD_column,"mean"),
            OD_std=(OD_column, "std"),
            n=(OD_column, "count"),
            mu_mean=(mu_column,"mean"),
            mu_std=(mu_column,"std"),
        )
        .reset_index()
    )

    summary_df["OD_se"] = (
        summary_df["OD_std"] / summary_df["n"].pow(0.5)
    )

    summary_df["OD_ci"] = summary_df.apply(
        lambda row: stats.t.ppf(0.975, row["n"] -1) * row["OD_se"]
        if row["n"] > 1 else 0,
        axis=1
    )

    summary_df["mu_se"] = (
            summary_df["mu_std"] / summary_df["n"].pow(0.5)
            .dropna()
    )

    summary_df["mu_ci"] = summary_df.apply(
        lambda row: stats.t.ppf(0.975, row["n"] - 1) * row["mu_se"]
        if row["n"] > 1 else 0,
        axis=1
    )

    return summary_df


# ---------------------------------------------------------
# full pipeline: combined TECAN data -> processed summary
# ---------------------------------------------------------

def process_tecans(data_df, metadata_df, assignment_df, **kwargs):
    """
    Run the whole Dataprocessing pipeline on imported TECAN data.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Combined wide-format TECAN data (from Dataimport.concatenate_tecan_files).

    metadata_df : pandas.DataFrame
        Combined TECAN metadata (only used for logging/consistency).

    assignment_df : pandas.DataFrame
        Well assignment sheet (columns: well, condition, replicate + extras).

    Returns
    -------
    summary_df : pandas.DataFrame
        Summarised mean / se / growth-rate data ready for Plot_results.
    """
    check_well_columns(data_df)

    # The TECAN import can leave the time column as object dtype; coerce it to
    # a numeric value so the growth-rate math below is safe (and reproducible).
    time_column = find_time_column(data_df)
    data_df = data_df.copy()
    data_df[time_column] = pd.to_numeric(data_df[time_column], errors="coerce")

    well_assignments = read_well_assignments(assignment_df)

    long_df = assign_conditions(data_df, well_assignments)

    filtered_df = remove_spikes(
        long_df,
        window=kwargs.get("window", 20),
        threshold=kwargs.get("threshold", 2)
    )

    smoothed_df = (
        filtered_df
        .groupby("well", group_keys=False)
        .apply(smooth_with_savgol)
        .sort_index()  # restore original row order (apply can reorder groups)
    )

    # Newer pandas (>= 3.0) drops the grouping column from the apply output.
    # Re-attach it (index-aligned) so the next groupby on "well" keeps working.
    if "well" not in smoothed_df.columns:
        smoothed_df["well"] = filtered_df["well"]

    mu_df = (
        smoothed_df
        .groupby(["well"], group_keys=False)
        .apply(calculate_growthrate)
    )

    summary_df = calculate_summary(
        mu_df,
        OD_column="Y Data smoothed",
        mu_column="mu",
        well_assignments=well_assignments,
    )

    return summary_df


if __name__ == "__main__":

    BASE = Path(r"C:\Users\rklausser\Programming\20261109_Pseudomonas_OD\Tecan-OD600-growthrate\Main")
    data_df = pd.read_parquet(BASE / "Temp" / "tecan_import_data_df.parquet")
    metadata_df = pd.read_parquet(BASE / "Temp" / "tecan_import_meta_df.parquet")
    assignment_df = pd.read_excel(
        r"C:\Users\rklausser\Programming\20261109_Pseudomonas_OD\Tecan-OD600-growthrate\Well_assignment\assignment_sheet.xlsx"
    )

    summary_df = process_tecans(data_df, metadata_df, assignment_df)
    print(summary_df.head()) #optional diagnostic
    summary_df.to_parquet(BASE / "Temp" / "summary_df.parquet")
