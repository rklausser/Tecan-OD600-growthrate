import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from dataclasses import dataclass, field
from itertools import product
from math import ceil
from pathlib import Path

# --------------------------
# file import
# --------------------------

summary_df = pd.read_parquet(r"C:\Users\rklausser\Programming\20261109_Pseudomonas_OD\Tecan-OD600-growthrate\Main\Temp\summary_df.parquet")

#Metadata import is not used at the moment. Can be reactivated later
#metadata_df = pd.read_parquet(r"C:\Users\rklausser\Programming\20261109_Pseudomonas_OD\Tecan-OD600-growthrate\Main\Temp\tecan_import_meta_df.parquet")


# Dictionaries to define standard lists for lines and colors

LINESTYLES = ['-', '--', '-.', ':']

COLOR_BASE = ['#332288','#117733','#44aa99', "#88ccee",
              "#ddcc77", "#cc6677", "#aa4499", "#882255"]

BASE = Path(__file__).resolve().parent

# ---------------------------------------------------------
# Config definition
# ---------------------------------------------------------

@dataclass
class PlotConfig:
    color_var: str | None = None       # tickbox choice 1, what is displayed as color?
    color_order: list[str] = field(default_factory=lambda: COLOR_BASE[:])
    style_var: str | None = None   # tickbox choice 2, what is displayed as linestyles?
    style_order: list[str] = field(default_factory=lambda: LINESTYLES[:])
    facet_vars: list[str] = field(default_factory=list)  # any remaining params will be shown as facets
    y_mean: str = "OD_mean"     # unspecific definition of the y-data so one function can plot both OD and mu
    y_se: str = "OD_se"         # unspecific definition of the y-data so one function can plot both OD and mu
    ylabel: str = "OD600"       # unspecific definition of the y-data so one function can plot both OD and mu
    ncol: int = 3               # sets max. columns of the panel before they wrap
    filters: dict[str, list] = field(default_factory=dict) # empty produced no filter
    


# ---------------------------------------------------------
# Helper function definitions
# ---------------------------------------------------------

METRIC_COLUMNS = {"OD_mean", "OD_std", "n", "mu_mean", "mu_std", "OD_se", "OD_ci", "mu_se", "mu_ci"} # all the metrics calculated by Dataprocessing.py

def parameter_columns(df):
    """returns all parameter columns that are selectable for plotting"""
    return [c for c in df.columns
            if c not in METRIC_COLUMNS and "Time" not in str(c)]

def unique(df, var):
    return sorted(pd.unique(df[var].dropna())) # sorts all entries for a parameter column to return the full sorted range of parameter values as a tuple

def style_map(df, cfg):
    vals = unique(df, cfg.style_var)
    return {v: cfg.style_order[i % len(cfg.style_order)] for i, v in enumerate(vals)} # returns the linestyle order for the specific parameter values in each plot

def color_map(df, cfg):
    vals = unique(df, cfg.color_var)
    return {v: cfg.color_order[i % len(cfg.color_order)] for i, v in enumerate(vals)} # returns the color order for the specific parameter values in each plot

def facet_combos(df, vars_):
    if not vars_:
        return[()]
    return list(product(*[unique(df, v) for v in vars_])) # creates the Cartesian product of all parameter value combinations in vars_

def default_config(df, y_mean="OD_mean", y_se="OD_se", ylabel="OD600", **kwargs):
    params = parameter_columns(df)
    cfg = PlotConfig(y_mean=y_mean, y_se=y_se, ylabel=ylabel, **kwargs)
    if params:
        cfg.color_var = params[0]
    if len(params) > 1:
        cfg.style_var = params[1]
    return cfg

def apply_filters(df, filters):
    for col, vals in filters.items():
        df = df[df[col].isin(vals)]
    return df


    

# ---------------------------------------------------------
# Plotting the selected y data based on the selected config
# ---------------------------------------------------------

def plot_mean_metric(df, cfg):
    df = apply_filters(df, cfg.filters) # filters the data for selected parameter ranges
    style = style_map(df, cfg) # takes parameter to be line-styled from config
    color = color_map(df, cfg) # takes parameter to be color-styled from config
    combos = facet_combos(df, cfg.facet_vars) # takes list of remaining params that was shown in the config
    
    if combos:
        ncol = max(1, min(len(combos), cfg.ncol)) # is at least 1 so /0 divisions are avoided
        nrow = ceil(len(combos) / ncol)
    else:
        ncol, nrow = 1, 1 # when nothing is in combos, there is no need to wrap so it becomes a single panel
        # Note: Input fields are tricky in terms of safety, so ncol should be selected in the config via a spinner or something like that

    fig, axes = plt.subplots(nrow, ncol, squeeze=False, figsize=(10, 6))
    axes = axes.ravel()

    # Filter the data for each panel of the facets, only lets values for the respective variables through
    for ax, combo in zip(axes, combos):
        panel = df
        for var, val in zip(cfg.facet_vars, combo):
            panel = panel[panel[var] == val]

        for sval in unique(panel, cfg.style_var):
            for cval in unique(panel, cfg.color_var):
                sub = panel[(panel[cfg.style_var] == sval) & (panel[cfg.color_var] == cval)]
                sub = sub.dropna(subset=[cfg.y_mean]).sort_values("Time [s]")
                if sub.empty:
                    continue
                t = pd.to_numeric(sub["Time [s]"], errors="coerce") / 3600
                y = pd.to_numeric(sub[cfg.y_mean], errors="coerce")
                se = pd.to_numeric(sub[cfg.y_se], errors="coerce").fillna(0)

                ax.plot(t, y, linestyle=style[sval], linewidth=3, color=color[cval])
                ax.fill_between(t, y - se, y + se, color=color[cval], alpha=0.2)

                ax.set_xlabel("Time [h]"); ax.set_ylabel(cfg.ylabel); ax.grid(True)

# Generate the legend handles - this might need some more work?

    color_handles = [mlines.Line2D([], [], color=c, label=f"{cfg.color_var} = {v}", linestyle="-")
                     for v, c in color.items()]
    style_handles = [mlines.Line2D([], [], color="grey", label=f"{cfg.style_var} = {v}", linestyle=ls)
                     for v, ls in style.items()]
    fig.legend(handles=color_handles, title="Colors", loc="upper left")
    fig.legend(handles=style_handles, title="Lines", loc="upper right")

    return fig



# ---------------------------------------------------------
# Show and save Plot (should be insude the plotting function?)
# ---------------------------------------------------------

if __name__ == "__main__":

    od_cfg = default_config(summary_df)
    mu_cfg = default_config(summary_df, y_mean= "mu_mean", y_se="mu_se", ylabel="mu[h^-1]")

    od_fig = plot_mean_metric(summary_df, od_cfg)
    od_fig.tight_layout()
    od_fig.savefig(BASE / "Plots" / "OD.svg", transparent=True, format="svg")
    od_fig.savefig(BASE / "Plots" / "OD.png", transparent=False, format="png", dpi=600)

    mu_fig = plot_mean_metric(summary_df, mu_cfg)
    mu_fig.tight_layout()
    mu_fig.savefig(BASE / "Plots" / "mu.svg", transparent=True, format="svg")
    mu_fig.savefig(BASE / "Plots" / "mu.png", transparent=False, format="png", dpi=600)
 #   plt.show() # only turn on for directly running the script, not required in streamlit 
