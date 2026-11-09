#import required packages
import pandas as pd
from pathlib import Path

#define import directory
directory = Path(r"C:\Users\rklausser\Programming\20261109_Pseudomonas_OD\Tecan-OD600-growthrate\Test_Dataset")

#import meta-data function
def read_tecan(filepath, sheet_name="Result sheet"):
    """
    Read a TECAN Excel result sheet.

    Returns
    -------
    metadata_df : pandas.DataFrame
        Metadata as parameter/value pairs.

    data_df : pandas.DataFrame
        Kinetic measurement data, with one row per cycle.
    """

    filepath = Path(filepath)
    filename = Path(filepath).name #extract filename

    raw = pd.read_excel(
        filepath, # is replaced by the function argument for the filepath
        sheet_name=sheet_name, # "sheet_name" is equivalent to not using the specifier. Could be changed later
        header=None # prevents automatic interpretation of the top row as column header
    )

    #--------------------------
    # Find measurement header row
    #--------------------------

    header_matches = raw.index[
        raw.iloc[:, 0].astype(str).str.strip() == "Cycle Nr."
    ]
        # Explanations:
        # iloc searches based on position with syntax df.iloc[rows, columns]; : means "all rows"; 0 is column zero = 'A'
        # .astype(str) converts the content to strings > that way "Cycle Nr." can be correctly found.
        # .str.strip() removes whitespace at the beginning and end of the string > redundency/robustness

    if len(header_matches) == 0:
        raise ValueError(
            "Could not find 'Cycle Nr.' row in the provided file"
        )

    if len(header_matches) > 1:
        raise ValueError(
            "Multiple 'Cycle Nr.' rows found in the provided file"
        )

    header_row = header_matches[0]

    # --------------------------
    # Extract measurement data
    # --------------------------

    data_df = raw.iloc[header_row + 1:].copy() # copy measurement data into dataframe
    data_df.columns = raw.iloc[header_row] # set header row contents as column names

    data_df = data_df.dropna(how="all") # removes completely empty rows

    data_df = data_df[
        pd.to_numeric(
            data_df["Cycle Nr."],
            errors="coerce"
        ).notna()
    ].copy() # keeps only rows that contain a "Cycle Nr"
    data_df["Cycle Nr."] = data_df["Cycle Nr."].astype(int) # converts cycle numbers to integers

    data_df["Filename"] = filename

    # --------------------------
    # Extract meta data
    # --------------------------

    metadata = [] # create empty python LIST, it will become a list of dictionaries - hence the ({}) append construction later

    for row in range(header_row): # creates range from 0 to header_row

        parameter = raw.iloc[row, 0] # all entries in the rows range of column 0 (=A)
        value = raw.iloc[row, 4] # all entries in the rows range of column 4 (=E)

        if pd.isna(parameter) or pd.isna(value): # checks if parameter in row i is empty
            continue            # skips the rest of this iteration and moves to i+1

        parameter = str(parameter).strip() # removes whitespace

        if pd.isna(value):
            continue # same as with parameters

        metadata.append({
            "Parameter": parameter,
            "Value": value,
            "Filename": filename
        }) # This }) structure makes sure the list is appended with dictionaries -> metadata = [{}, {}, {}]


    metadata_df = pd.DataFrame(metadata) # This converts the Python list of dictionaries into a pandas DataFrame (apparently by using black magic)
    metadata_df["Value"] = metadata_df["Value"].astype(str) # converts all metadata to strings for consistency and saving

    return data_df, metadata_df


#------------------------------------------------------------
#data_df, metadata_df = read_tecan("20260805_RKRKL_P_knackmussii_imidazol.xlsx")

#print(data_df.to_string(index=False))
#print(metadata_df.to_string(index=False))

#data_df.to_parquet("tecan_import.parquet") # save data to parquet
#metadata_df.to_parquet("tecan_import_meta.parquet") # save metadata to parquet
#-------------------------------------------------------------


def get_metadata_value(metadata_df, parameter_name):
    matches = metadata_df.loc[
        metadata_df["Parameter"].eq(parameter_name)
    ]

    if matches.empty:
        raise KeyError(
            f"Metadata parameter {parameter_name!r} not found in tecan_import.parquet"
        )

    return matches.iloc[0]

def concatenate_tecan_files(
        filepaths,
        sheet_name="Result sheet",
        start_time_parameter="Start Time",
):
    imported_data = []
    imported_metadata = []

    for filepath in filepaths:
        data_df, metadata_df = read_tecan(filepath, sheet_name)

        start_time = metadata_df.loc[
        metadata_df["Parameter"] == start_time_parameter,
        "Value"
        ]

        if start_time.empty:
            raise ValueError(
                f"Could not find '{start_time_parameter}' in metadata."
        )

        start_time = start_time.iloc[0]

        print(f"Start time found: {start_time!r}")

        start_time = pd.to_datetime(start_time)

        imported_data.append({
            "data": data_df,
            "metadata": metadata_df,
            "start_time": start_time,
            "filepath": Path(filepath),
        })

    imported_data.sort(
        key = lambda item: item["start_time"]
    )

    combined_data = []
    time_offset = 0.0
    cycle_offset = 0

    for item in imported_data:
        data = item["data"].copy()

        data = data.sort_values("Time [s]").reset_index(drop=True)

        data["Original Cycle Nr."] = data["Cycle Nr."]

        data["Time [s]"] = (
                data["Time [s]"] + time_offset
        )

        data["Cycle Nr."] = (
                data["Cycle Nr."] + cycle_offset
        )

        combined_data.append(data)

        time_offset = data["Time [s]"].max()
        cycle_offset = data["Cycle Nr."].max()

    combined_data = pd.concat(
        combined_data,
        ignore_index = True
    )

    combined_metadata = pd.concat(
        [
            item["metadata"]
            for item in imported_data
        ],
        ignore_index = True
    )

    return combined_data, combined_metadata

filepaths = []
for file in directory.glob("*.xlsx"):
    filepaths.append(Path(file))

data_df, metadata_df = concatenate_tecan_files(
    filepaths,
    start_time_parameter="Start Time"
)

print(data_df.to_string(index=False))
print(metadata_df.to_string(index=False))

data_df.to_parquet(r"C:\Users\rklausser\Programming\20261109_Pseudomonas_OD\Tecan-OD600-growthrate\Main\Temp/tecan_import_data_df.parquet") # save data to parquet
metadata_df.to_parquet(r"C:\Users\rklausser\Programming\20261109_Pseudomonas_OD\Tecan-OD600-growthrate\Main\Temp/tecan_import_meta_df.parquet") # save metadata to parquet

