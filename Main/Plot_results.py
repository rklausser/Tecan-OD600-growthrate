import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# --------------------------
# file import
# --------------------------

summary_df = pd.read_parquet(r"C:\Users\rklausser\Programming\20261109_Pseudomonas_OD\Tecan-OD600-growthrate\Main\Temp\summary_df.parquet")
metadata_df = pd.read_parquet(r"C:\Users\rklausser\Programming\20261109_Pseudomonas_OD\Tecan-OD600-growthrate\Main\Temp\tecan_import_meta_df.parquet")

# --------------------------
# Plot settings
# --------------------------

# Select conditions to plot

imidazol_plot = [0, 0.5, 1, 1.5]
OD_plot = [0.1, 0.2, 0.5]
dummy_plot = [4]

# Dictionaries to define which parameters should be styled in which way
linestyle_dict = {
    0: '-',
    0.5: '--',
    1: '-.',
    1.5: ':'
}

color_dict = {
    0.1: '#1a85ff',
    0.2: '#a7b500',
    0.5: '#d41159'
}

print(summary_df)

plot_subset = summary_df[
        (summary_df["imidazole (g/L)"].isin(imidazol_plot)) &
        (summary_df["start OD"].isin(OD_plot)) &
        (summary_df["dummy_parameter"].isin(dummy_plot))
    ]


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

def plot_OD(plot_subset):
    """
    Plot mean ± standard error for each condition.
    """

    fig = plt.figure(figsize=(10, 6))

    for condition in plot_subset["condition"].unique():

        imidazole_conc = plot_subset.loc[plot_subset["condition"] == condition, "imidazole (g/L)"].to_numpy()[0]
        OD600_value = plot_subset.loc[plot_subset["condition"] == condition, "start OD"].to_numpy()[0]
        dummy_value = plot_subset.loc[plot_subset["condition"] == condition, "dummy_parameter"].to_numpy()[0]

        condition_data = plot_subset[
            plot_subset["condition"] == condition
        ].sort_values("Time [s]")

        condition_data = condition_data.dropna(
            subset=["Time [s]", "OD_mean", "OD_ci"]
        )

        if len(condition_data) == 0:
            continue

        time =  pd.to_numeric(condition_data["Time [s]"], errors="coerce") / 3600
        mean = pd.to_numeric(condition_data["OD_mean"], errors="coerce")
        se = pd.to_numeric(condition_data["OD_se"], errors="coerce").fillna(0)

        plt.plot(
            time,
            mean,
            linestyle=imi_linestyle_dict[imidazole_conc],
            linewidth=3,
            color=OD_color_dict[OD600_value],
            label=condition,
        )

        plt.fill_between(
            time,
            mean - se,
            mean + se,
            color=OD_color_dict[OD600_value],
            alpha=0.2
        )



    processed_files = []
    for item in metadata_df["Filename"].unique():
        processed_files.append(item)
        processed_files.append("\n")
    titlestring = ''.join(processed_files)



    plt.xlabel("Time [h]")
    plt.ylabel("OD600")
    plt.title(f"Processed files: \n {titlestring}") #storing data complicates the falename issue

    color_handles = [mlines.Line2D(
        [],
        [],
        color=OD_color_dict[OD600_value],
        label=f"start OD600 = {OD600_value}",
        linestyle="-"
    )
    for OD600_value in OD_plot]

    line_handles = [mlines.Line2D(
        [],
        [],
        color="grey",
        label=f"imidazole [g/L] = {imidazole_conc}",
        linestyle=imi_linestyle_dict[imidazole_conc],
    )
    for imidazole_conc in imidazol_plot]

    Legend_1 = plt.legend(
        handles=color_handles,
        title="Colors",
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.95),
    )
    plt.gca().add_artist(Legend_1)

    # Force Matplotlib to calc. dimensions of the Leend
    fig.canvas.draw()

    #Get width of Legend_1 in coordinates of the figure
    bbox = Legend_1.get_window_extent()
    bbox = bbox.transformed(fig.transFigure.inverted())

    Legend_1_height = bbox.height


    Legend_2 = plt.legend(
        handles=line_handles,
        title="Lines",
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.95 - Legend_1_height*1.5),
    )


    plt.grid(True)

    plt.tight_layout()

    plt.savefig(r"C:\Users\rklausser\Programming\20261109_Pseudomonas_OD\Tecan-OD600-growthrate\Main\Plots\OD.svg", transparent=True, format="svg")
    plt.savefig(r"C:\Users\rklausser\Programming\20261109_Pseudomonas_OD\Tecan-OD600-growthrate\Main\Plots\OD.png", transparent=None, format="png", dpi=600)
    plt.show()





def plot_mu(plot_subset):
    """
    Plot mean ± standard error for each condition.
    """

    fig = plt.figure(figsize=(10, 6))

    for condition in plot_subset["condition"].unique():

        imidazole_conc = plot_subset.loc[plot_subset["condition"] == condition, "imidazole (g/L)"].to_numpy()[0]
        OD600_value = plot_subset.loc[plot_subset["condition"] == condition, "start OD"].to_numpy()[0]

        condition_data = plot_subset[
            plot_subset["condition"] == condition
        ].sort_values("Time [s]")

        condition_data = condition_data.dropna(
            subset=["Time [s]", "mu_mean", "mu_ci"]
        )

        if len(condition_data) == 0:
            continue

        time =  pd.to_numeric(condition_data["Time [s]"], errors="coerce") / 3600
        mean = pd.to_numeric(condition_data["mu_mean"], errors="coerce")
        se = pd.to_numeric(condition_data["mu_se"], errors="coerce").fillna(0)

        plt.plot(
            time,
            mean,
            linestyle=imi_linestyle_dict[imidazole_conc],
            linewidth=3,
            color=OD_color_dict[OD600_value],
            label=condition,
        )

        plt.fill_between(
            time,
            mean - se,
            mean + se,
            color=OD_color_dict[OD600_value],
            alpha=0.2
        )



    processed_files = []
    for item in metadata_df["Filename"].unique():
        processed_files.append(item)
        processed_files.append("\n")
    titlestring = ''.join(processed_files)



    plt.xlabel("Time [h]")
    plt.ylabel(f"growth rate [h{{$^-1$}}]")
    plt.title(f"Processed files: \n {titlestring}") #storing data complicates the falename issue

    color_handles = [mlines.Line2D(
        [],
        [],
        color=OD_color_dict[OD600_value],
        label=f"start OD600 = {OD600_value}",
        linestyle="-"
    )
    for OD600_value in OD_plot]

    line_handles = [mlines.Line2D(
        [],
        [],
        color="grey",
        label=f"imidazole [g/L] = {imidazole_conc}",
        linestyle=imi_linestyle_dict[imidazole_conc],
    )
    for imidazole_conc in imidazol_plot]

    Legend_1 = plt.legend(
        handles=color_handles,
        title="Colors",
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.95),
    )
    plt.gca().add_artist(Legend_1)

    # Force Matplotlib to calc. dimensions of the Leend
    fig.canvas.draw()

    #Get width of Legend_1 in coordinates of the figure
    bbox = Legend_1.get_window_extent()
    bbox = bbox.transformed(fig.transFigure.inverted())

    Legend_1_height = bbox.height


    Legend_2 = plt.legend(
        handles=line_handles,
        title="Lines",
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.95 - Legend_1_height*1.5),
    )


    plt.grid(True)

    plt.tight_layout()

    plt.savefig(r"C:\Users\rklausser\Programming\20261109_Pseudomonas_OD\Tecan-OD600-growthrate\Main\Plots\mu.svg", transparent=True, format="svg")
    plt.savefig(r"C:\Users\rklausser\Programming\20261109_Pseudomonas_OD\Tecan-OD600-growthrate\Main\Plots\mu.png", transparent=None, format="png", dpi=600)
    plt.show()


plot_OD(plot_subset)
plot_mu(plot_subset)