import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter


# --------------------------
# file import
# --------------------------

data_df = pd.read_parquet("Temp/tecan_import_data_df.parquet")
metadata_df = pd.read_parquet("Temp/tecan_import_meta_df.parquet")


# --------------------------
# identify time column
# --------------------------

time_columns = [
    column
    for column in data_df.columns
    if "Time" in str(column)
]

if len(time_columns) == 0:
    raise ValueError("No time columns found")

if len(time_columns) > 1:
    raise ValueError(
        f"Multiple time columns found: {time_columns}; only 1 time column allowed"
    )

time_column = time_columns[0]

# --------------------------
# identify wells
# --------------------------

well_columns = [
    f"{row}{column}"
    for row in "ABCD"
    for column in range(1, 7)
]

# Check that all expected wells exist
missing_wells = [
    well
    for well in well_columns
    if well not in data_df.columns
]

if missing_wells:
    raise ValueError(
        f"Missing well columns: {missing_wells}"
    )

# ---------------------------------------------------------
# Well assignment
# ---------------------------------------------------------

well_assignments = {
    "A1": {
        "condition": "Control 0.1",
        "replicate": 1,
        "imidazole (g/L)": 0,
        "start OD": 0.1,
    },
    "A2": {
        "condition": "Control 0.1",
        "replicate": 2,
        "imidazole (g/L)": 0,
        "start OD": 0.1,
    },
    "A3": {
        "condition": "Control 0.1",
        "replicate": 3,
        "imidazole (g/L)": 0,
        "start OD": 0.1,
    },
    "A4": {
        "condition": "Control 0.2",
        "replicate": 1,
        "imidazole (g/L)": 0,
        "start OD": 0.2,
    },
    "A5": {
        "condition": "Control 0.2",
        "replicate": 2,
        "imidazole (g/L)": 0,
        "start OD": 0.2,
    },
    "A6": {
        "condition": "Control 0.2",
        "replicate": 3,
        "imidazole (g/L)": 0,
        "start OD": 0.2,
    },
    "B1": {
        "condition": "I 0.5, OD 0.1",
        "replicate": 1,
        "imidazole (g/L)": 0.5,
        "start OD": 0.1,
    },
    "B2": {
        "condition": "I 0.5, OD 0.1",
        "replicate": 2,
        "imidazole (g/L)": 0.5,
        "start OD": 0.1,
    },
    "B3": {
        "condition": "I 0.5, OD 0.1",
        "replicate": 3,
        "imidazole (g/L)": 0.5,
        "start OD": 0.1,
    },
    "B4": {
        "condition": "I 1.0, OD 0.2",
        "replicate": 1,
        "imidazole (g/L)": 1,
        "start OD": 0.2,
    },
    "B5": {
        "condition": "I 1.0, OD 0.2",
        "replicate": 2,
        "imidazole (g/L)": 1,
        "start OD": 0.2,
    },
    "B6": {
        "condition": "I 1.0, OD 0.2",
        "replicate": 3,
        "imidazole (g/L)": 1,
        "start OD": 0.2,
    },
    "C1": {
        "condition": "I 1.0, OD 0.1",
        "replicate": 1,
        "imidazole (g/L)": 1,
        "start OD": 0.1,
    },
    "C2": {
        "condition": "I 1.0, OD 0.1",
        "replicate": 2,
        "imidazole (g/L)": 1,
        "start OD": 0.1,
    },
    "C3": {
        "condition": "I 1.0, OD 0.1",
        "replicate": 3,
        "imidazole (g/L)": 1,
        "start OD": 0.1,
    },
    "C4": {
        "condition": "Control 0.5",
        "replicate": 1,
        "imidazole (g/L)": 0,
        "start OD": 0.5,
    },
    "C5": {
        "condition": "Control 0.5",
        "replicate": 2,
        "imidazole (g/L)": 0,
        "start OD": 0.5,
    },
    "C6": {
        "condition": "Control 0.5",
        "replicate": 3,
        "imidazole (g/L)": 0,
        "start OD": 0.5,
    },
    "D1": {
        "condition": "I 1.5, OD 0.1",
        "replicate": 1,
        "imidazole (g/L)": 1.5,
        "start OD": 0.1,
    },
    "D2": {
        "condition": "I 1.5, OD 0.1",
        "replicate": 2,
        "imidazole (g/L)": 1.5,
        "start OD": 0.1,
    },
    "D3": {
        "condition": "I 1.5, OD 0.1",
        "replicate": 3,
        "imidazole (g/L)": 1.5,
        "start OD": 0.1,
    },
    "D4": {
        "condition": "I 1.0, OD 0.5",
        "replicate": 1,
        "imidazole (g/L)": 1,
        "start OD": 0.5,
    },
    "D5": {
        "condition": "I 1.0, OD 0.5",
        "replicate": 2,
        "imidazole (g/L)": 1,
        "start OD": 0.5,
    },
    "D6": {
        "condition": "I 1.0, OD 0.5",
        "replicate": 3,
        "imidazole (g/L)": 1,
        "start OD": 0.5,
    }
}

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
                f"Well {well} has no condition assigned."
            )

        if "replicate" not in assignment:
            raise ValueError(
                f"Well {well} has no replicate number."
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

    print("Well assignments validated successfully.")


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
    time_columns = [
        column
        for column in data_df.columns
        if "Time" in str(column)
    ]

    if len(time_columns) != 1:
        raise ValueError(
            f"Expected exactly one time column, "
            f"found: {time_columns}"
        )

    time_column = time_columns[0]

    # Convert wide → long
    long_df = data_df.melt(
        id_vars=["Cycle Nr.", time_column],
        value_vars=well_columns,
        var_name="Well",
        value_name="Y Data"
    )

    # Create assignment DataFrame
    assignment_df = pd.DataFrame.from_dict(
        well_assignments,
        orient="index"
    )

    assignment_df.index.name = "Well"

    assignment_df = assignment_df.reset_index()

    # Merge assignments onto the measurement data
    long_df = long_df.merge(
        assignment_df,
        on="Well",
        how="left"
    )

    return long_df


long_df = assign_conditions(data_df, well_assignments)
# print(long_df) #optional diagnostic
long_df.to_parquet("Temp/long_df.parquet") # save data to parquet


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
        .groupby("Well")["Y Data"]
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
        .groupby("Well")["absolute_deviation"]
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



filtered_df = remove_spikes(
    long_df,
    window=20,
    threshold=2
)
filtered_df.to_parquet("Temp/filtered_df.parquet") # save data to parquet

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

# Apply savgol filter to each well separately


smoothed_df = (
    filtered_df
    .groupby("Well", group_keys=True)
    .apply(smooth_with_savgol)
    .reset_index()
    .drop(columns="level_1")
)

smoothed_df.to_parquet("Temp/smoothed_df.parquet") # save smoothed df

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

mu_df = (
    smoothed_df
    .groupby(
        ["Well"],
        group_keys=True
    )
    .apply(calculate_growthrate)
    .reset_index()
    .drop(columns=["level_1"])
)

# Optional diagnostics

#print(smoothed_df.head())
#print(mu_df.head())

# ---------------------------------------------------------
# calculate summary of OD ( mean and se) before plotting
# ---------------------------------------------------------

def calculate_summary(mu_df, OD_column, mu_column):
    """
    Calculate mean, standard deviation, number of replicates,
    and standard error for each condition and time point.
    """

    summary_df = (
        mu_df
        .groupby(["Time [s]", "condition", "imidazole (g/L)", "start OD"])
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

summary_df = calculate_summary(mu_df, OD_column = "Y Data smoothed", mu_column = "mu")
#print(summary_df.head()) #optional diagnostic
summary_df.to_parquet("Temp/summary_df.parquet")
