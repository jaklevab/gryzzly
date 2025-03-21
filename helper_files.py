import pandas as pd
from tqdm import tqdm as tqdmn
import numpy as np
import sys
from collections import Counter
import re
from functools import reduce
from scipy.stats import zscore
import statsmodels.api as sm

# CCDF: Calculates the proportion of elements >= val for each value in the range [min(x_l), max(x_l)).
ccdf = lambda x_l: [(x_l >= val).sum() / len(x_l) for val in range(min(x_l), max(x_l))]

# Degree distribution: Counts unique y values grouped by x, sorted by frequency (desc).
degree_dist = lambda x, y: dict(
    sorted(Counter(upd_df_decs.groupby(x)[y].nunique().values).items(), key=lambda pair: pair[1], reverse=True)
)

# Parse dates: Converts strings to datetime; keeps floats unchanged.
parse_dates = lambda col: [
    pd.to_datetime(k.split()[0], format='%Y-%m-%d', errors="coerce") if not isinstance(k, float) else k
    for k in col
]

# Inter-event times (IET): Computes time differences (in days) between sorted events in x.
iet = lambda x: np.array([(y - z) / np.timedelta64(1, 'D') for rx in [np.sort(np.array(x))] for y, z in zip(rx[1:], rx[:-1])])

# Continuous CCDF: Calculates CCDF for a custom array of values.
cont_ccdf = lambda x_l, arr: [(x_l >= val).sum() / len(x_l) for val in arr]

# Burstiness: (standard deviation − mean) / (mean + standard deviation)
burstiness = lambda x: (np.std(x) - np.mean(x)) / (np.mean(x) + np.std(x))

# Union of intersection: size(intersection of sets) / size(union of sets)
uoi = lambda x: len(reduce(set.intersection, x)) / (len(reduce(set.union, x)) + 0.0)

# Helper to check values are not zero or NaN
not_null_zero = lambda x: (x != 0) & (~x.isna())

# Boolean filter for time-related columns
get_bool_time = lambda x: not_null_zero(x["planned_duration"]) & not_null_zero(x["elapsed_duration"])

# Boolean filter for budget-related columns
get_bool_budget = lambda x: not_null_zero(x["budget_amount"]) & not_null_zero(x["budget_spent"])

# Combined boolean filter for both time and budget
get_bool_time_budget = lambda x: get_bool_time(x) & get_bool_budget(x)

# Z-score normalization: (x - mean) / std
norm = lambda x: (x - x.mean()) / x.std()

# Outlier removal filter: keep data within [2.5%, 97.5%] quantiles
outl_rem = lambda x: (x < np.nanquantile(x, 0.975)) & (x > np.nanquantile(x, 0.025))

def load_declarations(data_dir):
    """
    Load and preprocess declaration data from the given directory.
    """
    # Load declarations data
    df_decs = pd.read_csv(data_dir + "declarations.csv")
    df_decs["fmt_crt"] = df_decs["created_at"].copy()
    
    # Extract time-related features
    df_decs["hour"] = [pd.to_datetime(k).hour for k in df_decs.fmt_crt]
    df_decs["year"] = [pd.to_datetime(k).year for k in df_decs.fmt_crt]
    df_decs["created_at"] = parse_dates(df_decs["created_at"])
    df_decs["declared_at"] = df_decs["date"].apply(lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors="coerce"))
    
    # Create additional features
    df_decs["str_dec"] = df_decs["declared_at"].astype(str)
    df_decs["duration"] = df_decs["duration"] / (3600 * 1e9)  # Convert duration to hours
    df_decs["edge_id"] = ["%s@%s" % (k, v) for k, v in df_decs[["user_id", "task_id"]].values]
    df_decs.drop(["date"], axis=1, inplace=True)
    
    # Merge with other datasets
    df_decs = df_decs\
        .merge(pd.read_csv(data_dir + "tasks.csv", usecols=["id", "project_id"]).rename({"id": "task_id"}, axis=1))\
        .merge(pd.read_csv(data_dir + "users.csv", usecols=["id", "team_id"]).rename({"id": "user_id"}, axis=1))\
        .merge(pd.read_csv(data_dir + "subscriptions.csv", usecols=["team_id"]))
    
    # Filter rows with valid dates and durations
    df_decs = df_decs[(~df_decs.created_at.isna()) & (~df_decs.declared_at.isna())]
    df_decs["str_dec"] = df_decs["declared_at"].astype(str)
    df_decs = df_decs[(df_decs.duration < 24) & (df_decs.duration > 0)]
    
    # Display basic statistics
    print("created_min", df_decs.created_at.min())
    print("created_max", df_decs.created_at.max())
    print("# users", df_decs.user_id.nunique())
    print("# tasks", df_decs.task_id.nunique())
    print("# projects", df_decs.project_id.nunique())
    print("# decs", df_decs.shape[0])
    print("# user-days", df_decs[["user_id", "created_at"]].drop_duplicates().shape[0])
    
    # Filter data based on date range
    min_date = pd.to_datetime("2017-09-01")
    max_date = pd.to_datetime(df_decs.created_at.max()) - pd.to_timedelta("90D")
    df_decs = df_decs[[min_date <= k <= max_date for k in tqdmn(df_decs.declared_at)]]
    
    return df_decs, max_date

def convert_to_hours(duration):
    """
    Convert a duration string (e.g., '38h30m0s') to hours as a float.
    """
    # Extract hours, minutes, and seconds from the string, defaulting to 0 if not present.
    hours = int(re.search(r"(\d+)h", duration).group(1)) if "h" in duration else 0
    minutes = int(re.search(r"(\d+)m", duration).group(1)) if "m" in duration else 0
    seconds = int(re.search(r"(\d+)s", duration).group(1)) if "s" in duration else 0
    # Calculate total hours.
    return hours + minutes / 60 + seconds / 3600


def pdf_dist(
    df_decs, x, y, nbins=20, noise_fraction=0.0, min_noise=0.0, min_count_tail=2
):
    """
    Calculate a log-binned probability density function (PDF) for a column grouped by another,
    with bins adjusted to integer boundaries to avoid empty bins due to the discrete (integer) nature
    of the distribution. The centers of each bin are given random noise to prevent overlapping points
    when multiple points fall at the same value.

    After binning:
      1) Trailing bins with zero counts are removed.
      2) If the total count in the final bins is below 'min_count_tail', they are merged into a single bin.
         This avoids a 'spiky' or fragmented tail (e.g., alternating 0 and 1 counts).

    Parameters
    ----------
    df_decs : pandas.DataFrame
        The input dataframe.
    x : str
        The column name to group by.
    y : str
        The column for which to compute unique counts per group.
    nbins : int, optional
        The desired number of logarithmic bins (default is 20).
    noise_fraction : float, optional
        Fraction of the bin width to use as noise amplitude (default is 0.0).
    min_noise : float, optional
        The minimum noise amplitude to ensure sufficient jitter (default is 0.0).
    min_count_tail : int, optional
        If the total counts in the tail bins (from the last bin backward)
        is below this threshold, merge them into one bin (default is 2).

    Returns
    -------
    bin_centers : np.ndarray
        The (noisy) bin centers after any removals/merging.
    pdf_values : np.ndarray
        The normalized probability density function values for each bin.
    edges : np.ndarray
        The final (integer) bin edges.
    """
    # 1) Group data and compute unique counts
    x_l = df_decs.groupby(x)[y].apply(lambda s: len(set(s))).values
    # Filter out zeros (log(0) undefined in log-binning)
    x_l = x_l[x_l > 0]
    if len(x_l) == 0:
        return None, None, None

    # 2) Create integer bin edges via log spacing
    min_val = int(x_l.min())
    max_val = int(x_l.max())

    raw_bins = np.logspace(np.log10(min_val), np.log10(max_val), nbins + 1)
    int_bins = np.unique(np.floor(raw_bins).astype(int))
    if int_bins[-1] <= max_val:
        int_bins = np.append(int_bins, max_val + 1)

    # 3) Compute histogram with these integer bins
    counts, edges = np.histogram(x_l, bins=int_bins)

    # 4) Compute bin centers and add random noise
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]
    noise_amplitude = np.maximum(noise_fraction * widths, min_noise)
    noise = np.random.uniform(-noise_amplitude, noise_amplitude)
    bin_centers_noisy = bin_centers + noise

    # ----------------------------------------------------------------------
    # 5) Remove trailing empty bins
    # ----------------------------------------------------------------------
    while len(counts) > 0 and counts[-1] == 0:
        counts = counts[:-1]
        edges = edges[:-1]
        bin_centers_noisy = bin_centers_noisy[:-1]

    if len(counts) == 0:
        return None, None, None

    # ----------------------------------------------------------------------
    # 6) Merge tail bins with low counts
    #    Merge bins from the tail until the cumulative count exceeds 'min_count_tail'
    # ----------------------------------------------------------------------
    tail_sum = 0
    tail_start = len(counts) - 1  # index from which we merge the tail
    while tail_start >= 0 and tail_sum < min_count_tail:
        tail_sum += counts[tail_start]
        tail_start -= 1

    merged_length = len(counts) - (tail_start + 1)
    if merged_length > 1:
        if tail_start < 0:
            # All bins merge into one single bin
            merged_bin_left = edges[0]
            merged_bin_right = edges[-1]
            new_counts = np.array([tail_sum])
            new_edges = np.array([merged_bin_left, merged_bin_right])
            new_centers = np.array([0.5 * (merged_bin_left + merged_bin_right)])
        else:
            # Merge the tail bins
            new_counts = counts[: tail_start + 1]
            new_edges = edges[: tail_start + 2]
            new_centers = bin_centers_noisy[: tail_start + 1]
            merged_bin_left = edges[tail_start + 1]
            merged_bin_right = edges[-1]
            new_edges[-1] = merged_bin_right
            merged_bin_center = 0.5 * (merged_bin_left + merged_bin_right)
            new_counts[-1] += tail_sum
            new_centers[-1] = merged_bin_center

        counts, edges, bin_centers_noisy = new_counts, new_edges, new_centers

    # 7) Normalize counts to obtain the probability density function
    widths_final = np.diff(edges)
    total_counts = counts.sum()
    pdf_values = counts / total_counts / widths_final

    return bin_centers_noisy, pdf_values, edges


def ser2df(ser, col_name="", idx_name=0):
    """
    Convert a pandas Series to a DataFrame with optional column and index names.
    """
    return pd.DataFrame(ser).reset_index().rename({idx_name: col_name}, axis=1)


def ccdf_sum_dist(df_decs, x, y):
    """
    Compute CCDF for the sum of durations grouped by two columns.
    """
    # Calculate the total duration for each group of (x, y) and flatten the values.
    x_l = np.array(df_decs.groupby([x, y])[["duration"]].sum().values).ravel()
    # Generate a range of values for the CCDF.
    x_p = np.linspace(0.001, 200, num=1000)
    # Calculate the proportion of values >= each value in x_p.
    y_l = [(x_l >= val).sum() / len(x_l) for val in x_p]
    return x_p, y_l


def get_zero_sequences_lengths(lst):
    """
    Get lengths of consecutive zero sequences in a list.
    """
    lengths = []
    count = 0
    # Iterate through the list and count consecutive zeros.
    for num in lst:
        if num == 0:
            count += 1
        else:
            if count > 0:
                lengths.append(count)  # Add count if a zero sequence ends.
                count = 0
    # Append the last count if the list ends with zeros.
    if count > 0:
        lengths.append(count)
    return lengths


def failure_streak_length(df_decs, df_object, grp_name, obj_name, date_name, random=False):
    """
    Calculate the lengths of failure streaks for grouped objects.
    """
    # Merge failure data with group and object info, sorting by date.
    cons_failure = df_decs[[grp_name, obj_name]].drop_duplicates()\
        .merge(df_object.sort_values(date_name)[[obj_name, "failure"]].drop_duplicates().dropna())\
        .groupby(grp_name)["failure"].apply(lambda x: list(x.values))
    
    # Optionally randomize failure data.
    if random:
        cons_failure = cons_failure.apply(lambda x: (np.random.random(size=len(x)) > 0.5).astype(int))
    
    # Compute lengths of zero sequences in failure streaks.
    cons_failure = cons_failure.apply(get_zero_sequences_lengths)
    # Flatten the list of lengths and return as an array.
    return np.array([j for k in cons_failure.values for j in k])

def generate_task_data(upd_df_decs, max_date, up_DATA_DIR):
    """
    Generate project-level data with metrics and statuses based on input data.
    """
    # Load and merge task details and computed metrics
    df_tasks = pd.read_csv(up_DATA_DIR + "tasks.csv").rename({"id": "task_id"}, axis=1)
    df_tasks_metrics = pd.read_csv(up_DATA_DIR + "tasks_computed.csv").rename({"id": "task_id"}, axis=1)
    df_tasks = df_tasks.merge(df_tasks_metrics[["task_id", "planned_duration", "elapsed_duration"]], on="task_id")
    
    # Convert durations to hours and compute evaluation scores
    df_tasks['planned_duration'] = df_tasks['planned_duration'].apply(convert_to_hours)
    df_tasks['elapsed_duration'] = df_tasks['elapsed_duration'].apply(convert_to_hours)
    df_tasks["eval_score"] = (1.0 - df_tasks["elapsed_duration"]) / (df_tasks["planned_duration"] + 0.0)

    # Filter tasks present in the declarations
    df_tasks = df_tasks.merge(upd_df_decs[["task_id"]].drop_duplicates())

    # Add min and max declaration dates for each task
    df_tasks = df_tasks.merge(upd_df_decs.groupby("task_id")[["declared_at"]].max()
                              .rename({"declared_at": "max_day_date"}, axis=1).reset_index(), on="task_id")
    df_tasks = df_tasks.merge(upd_df_decs.groupby("task_id")[["declared_at"]].min()
                              .rename({"declared_at": "min_day_date"}, axis=1).reset_index(), on="task_id")

    # Aggregate daily durations for each task
    df_day_task = upd_df_decs.sort_values("declared_at")\
                             .groupby(["task_id", "declared_at"])["duration"].sum().reset_index()

    # Compute cumulative daily durations
    res = df_day_task.groupby("task_id").apply(
        lambda group: list(zip(group["declared_at"], np.cumsum(group["duration"])))).reset_index(level=0)
    df_day_task = pd.DataFrame([(k.task_id, d[0], d[1]) for _, k in res.iterrows() for d in k[0]],
                               columns=["task_id", "declared_at", "dur_amount"]).merge(df_tasks)

    # Identify failure dates (first date cumulative duration meets or exceeds planned duration)
    df_tasks = df_tasks.merge(
        df_day_task[df_day_task.dur_amount >= df_day_task.planned_duration].groupby("task_id")[["declared_at"]].min()
        .reset_index().rename({"declared_at": "failure_date"}, axis=1), how="left")

    # Assign failure date as max declaration date if no failure occurred
    df_tasks.loc[df_tasks.failure_date.isna(), "failure_date"] = df_tasks.loc[df_tasks.failure_date.isna(), "max_day_date"]

    # Determine active status (active if max declaration date is within the last 90 days)
    df_tasks["is_active"] = [(k.days <= 90) for k in max_date - df_tasks["max_day_date"]]

    # Initialize success column and compute success based on duration metrics
    df_tasks["success"] = np.nan
    df_tasks["time_success_val"] = df_tasks.planned_duration - df_tasks.elapsed_duration
    df_tasks.loc[
        (df_tasks.planned_duration != 0) & (df_tasks.elapsed_duration != 0) &
        (df_tasks.planned_duration <= df_tasks.elapsed_duration), "success"] = False
    df_tasks.loc[
        (df_tasks.planned_duration != 0) & (df_tasks.elapsed_duration != 0) &
        (df_tasks.is_active == False) &
        (df_tasks.planned_duration > df_tasks.elapsed_duration), "success"] = True

    return df_tasks


def generate_project_data(upd_df_decs, max_date, up_DATA_DIR):
    """
    Generate project-level data with metrics and statuses based on input data.
    """
    # Load project details and metrics
    df_projects = pd.read_csv(up_DATA_DIR + "projects.csv").rename({"id": "project_id"}, axis=1)
    df_projects_metrics = pd.read_csv(up_DATA_DIR + "projects_computed.csv").rename({"id": "project_id"}, axis=1)

    # Convert durations to hours and compute an evaluation score
    df_projects_metrics['planned_duration'] = df_projects_metrics['planned_duration'].apply(convert_to_hours)
    df_projects_metrics['elapsed_duration'] = df_projects_metrics['elapsed_duration'].apply(convert_to_hours)
    df_projects_metrics = df_projects_metrics[df_projects_metrics['planned_duration'] > 0]
    df_projects_metrics["eval_score"] = (1.0 - df_projects_metrics["elapsed_duration"]) / df_projects_metrics["planned_duration"]

    # Merge project data with metrics
    df_projects = df_projects.merge(
        df_projects_metrics[["project_id", "planned_duration", "elapsed_duration"]],
        on="project_id"
    )

    # Aggregate daily durations for each project
    df_day_proj = (
        upd_df_decs.sort_values("declared_at")
        .groupby(["project_id", "declared_at"])["duration"]
        .sum()
        .reset_index()
    )

    # Compute cumulative duration over time
    res = df_day_proj.groupby("project_id").apply(
        lambda g: list(zip(g["declared_at"], np.cumsum(g["duration"])))
    ).reset_index(level=0)

    # Flatten cumulative data into a single DataFrame
    df_day_proj = pd.DataFrame(
        [(row.project_id, d[0], d[1]) for _, row in res.iterrows() for d in row[0]],
        columns=["project_id", "declared_at", "dur_amount"]
    ).merge(df_projects)

    # Merge additional date info
    df_projects = pd.merge(df_projects, upd_df_decs[["project_id"]].drop_duplicates())
    df_projects = df_projects.merge(
        upd_df_decs.groupby("project_id")["declared_at"].max().rename("max_day_date").reset_index(),
        on="project_id"
    )
    df_projects = df_projects.merge(
        upd_df_decs.groupby("project_id")["declared_at"].min().rename("min_day_date").reset_index(),
        on="project_id"
    )

    # Determine failure dates (when dur_amount >= planned_duration)
    df_projects = df_projects.merge(
        df_day_proj[df_day_proj.dur_amount >= df_day_proj.planned_duration]
        .groupby("project_id")["declared_at"].min()
        .rename("failure_date")
        .reset_index(),
        how="left",
        on="project_id"
    )
    # If no failure, set failure_date to max_day_date
    df_projects.loc[df_projects["failure_date"].isna(), "failure_date"] = \
        df_projects.loc[df_projects["failure_date"].isna(), "max_day_date"]

    # Mark projects as active if last declaration was within 90 days of max_date
    df_projects["is_active"] = [(max_date - d).days <= 90 for d in df_projects["max_day_date"]]

    # Initialize success-related columns
    df_projects["success"] = np.nan
    df_projects["time_success_val"] = df_projects["planned_duration"] - df_projects["elapsed_duration"]

    # Mark projects that exceed planned duration as failures
    df_projects.loc[
        (df_projects["planned_duration"] != 0) &
        (df_projects["elapsed_duration"] != 0) &
        (df_projects["planned_duration"] <= df_projects["elapsed_duration"]),
        "success"
    ] = False

    # Mark non-active projects that remain under planned duration as successes
    df_projects.loc[
        (df_projects["planned_duration"] != 0) &
        (df_projects["elapsed_duration"] != 0) &
        (df_projects["is_active"] == False) &
        (df_projects["planned_duration"] > df_projects["elapsed_duration"]),
        "success"
    ] = True

    return df_projects, df_day_proj


def generate_task_failure_score(df_tasks, df_projects):
    """
    Generate task failure scores for projects,
    focusing on the first and last failures.
    """

    # Filter tasks with known success status, sort by earliest declaration
    glb = df_tasks[~df_tasks.success.isna()]\
        .sort_values("min_day_date")[["task_id","project_id","eval_score"]]\
        .groupby("project_id")["eval_score"].apply(np.array)\
        .apply(lambda x: x[x <= 0])  # Consider only negative or zero scores
    
    # Extract the first and last negative score (if any) for each project
    glb = pd.concat(
        [
            glb.apply(lambda x: x[0] if len(x) > 0 else np.nan),
            glb.apply(lambda x: x[-1] if len(x) > 2 else np.nan)
        ],
        axis=1
    )
    glb.columns = ["First", "Last"]

    # Merge with project success status
    glb_plot = glb[~glb.First.isna()].reset_index()\
        .merge(df_projects[~df_projects.success.isna()][["project_id", "success"]].drop_duplicates())

    # Filter and reshape data for plotting
    glb_arr = glb_plot[(glb_plot.First >= -2) & (glb_plot.First <= 2)].dropna()\
        .melt(id_vars=["project_id", "success"], value_vars=["First", "Last"])
    glb_arr.columns = ["project_id", "Project Success", "Task Failure", "Task Score"]
    return glb_arr


def generate_task_failure_learning(df_tasks, df_projects):
    """
    Generate a DataFrame capturing inter-event times (IET) between consecutive failures,
    normalized based on the first failure's average delay for successful vs. failed projects.
    """
    
    # Group tasks with negative eval_score (failures), then compute inter-event times
    fail_iet = df_tasks[df_tasks.eval_score <= 0].sort_values("min_day_date")\
        .groupby(["project_id"])["min_day_date"]\
        .apply(iet).apply(np.array).reset_index()\
        .rename({"min_day_date": "tsk_iet"}, axis=1)\
        .merge(df_projects[~df_projects.success.isna()][["project_id", "success"]])

    # Flatten data: (failure sequence number, IET, project success)
    fail_iet = pd.DataFrame(
        [
            (1 + i, ie, s)
            for s, iet_lst in fail_iet[["success", "tsk_iet"]].values
            for i, ie in enumerate(iet_lst)
            if not np.isnan(ie) and not np.isinf(ie)
        ],
        columns=["nb_failed_tasks", "Tn", "success"]
    )

    # Normalize Tn by the average first-failure IET within each success category
    fail_iet.loc[fail_iet.success == False, "Tn"] /= (
        np.mean(fail_iet[(fail_iet.nb_failed_tasks == 1) & (fail_iet.success == False)]["Tn"]) + 0.0
    )
    fail_iet.loc[fail_iet.success == True, "Tn"] /= (
        np.mean(fail_iet[(fail_iet.nb_failed_tasks == 1) & (fail_iet.success == True)]["Tn"]) + 0.0
    )

    # Calculate aggregated statistics by failure count
    fail_iet_tot = fail_iet.groupby(['success', 'nb_failed_tasks']).agg(
        mean_Tn=('Tn', 'mean'),
        count=('Tn', 'count'),
        std_Tn=('Tn', 'std')
    ).reset_index()

    # Compute standard error
    fail_iet_tot['se_Tn'] = fail_iet_tot['std_Tn'] / np.sqrt(fail_iet_tot['count'])

    # Limit the analysis to the first 5 failures
    fail_iet_tot = fail_iet_tot[fail_iet_tot.nb_failed_tasks <= 5]

    return fail_iet_tot

# Convert a Series to a DataFrame, renaming the index column if desired.
def ser2df(ser, col_name="", idx_name=0):
    return (
        pd.DataFrame(ser)
        .reset_index()
        .rename({idx_name: col_name}, axis=1)
    )

# Compute day intervals (in whole days) between consecutive datetime values.
def get_day_interval(arr):
    # For each consecutive pair (previous, following), calculate (following - previous).days
    return [(f - p).days for p, f in zip(arr.iloc[:-1], arr.iloc[1:])]

def log_cols(data, cols):
    """
    Create log-transformed columns for specified numeric columns.
    If a column contains zeros, use log(1 + x) instead of log(x).
    """
    for k in cols:
        # Check if any values in column k are zero
        if (data[k] == 0).any():
            # log1p_<col> = ln(1 + col)
            data[f"log1p_{k}"] = np.log1p(data[k])
        else:
            # log_<col> = ln(col)
            data[f"log_{k}"] = np.log(data[k])
    return data

# Remove outliers in specified columns by filtering rows between two quantiles.
def remove_outliers(data, cols, low_qt=0, high_qt=1):
    # Define two lambda functions for filtering lower and upper bounds.
    outl_rem_low  = lambda x: x >= np.nanquantile(x, low_qt)
    outl_rem_high = lambda x: x <= np.nanquantile(x, high_qt)
    
    # Collect boolean masks for each column.
    rem_idxs = []
    for col in cols:
        rem_idxs.append(outl_rem_low(data[col]) & outl_rem_high(data[col]))
    
    # Combine all masks with a logical AND to keep only rows within the quantile range for all columns.
    inn = rem_idxs[0]
    for mask in rem_idxs[1:]:
        inn &= mask
    
    # Return the filtered DataFrame.
    return data[inn]

def tasks_per_project(df_decs):
    """
    Computes:  The total number of unique tasks per project.
    """

    # 1) Number of unique tasks per project
    tsks_per_prj = ser2df(
        df_decs.groupby("project_id")["task_id"].apply(lambda x: len(set(x))),
        col_name="nb_tasks",
        idx_name="task_id"
    )

    return tsks_per_prj
    
    
def decs_stats_project(df_decs):
    """
    Computes:
      1) Average declaration duration per project
      2) Average daily declaration duration per project
    """

    # 1) Average declaration duration per project
    avg_dec_tm_per_prj = (
        df_decs
        .groupby("project_id")[["duration"]]
        .mean()
        .reset_index()
    )

    # 2) Average daily declaration duration per project
    avg_dly_dec_tm_per_prj = (
        df_decs
        .groupby(["project_id", "task_id", "declared_at"])[["duration"]]
        .sum()
        .reset_index()
        .groupby("project_id")[["duration"]]
        .mean()
        .reset_index()
        .rename({"duration": "daily_duration"}, axis=1)
    )

    return avg_dec_tm_per_prj, avg_dly_dec_tm_per_prj


def users_per_project(df_decs):
    """
    Computes: The total number of unique users per project
    """

    # Number of unique users per project
    nb_wkrs_per_prj = ser2df(
        df_decs.groupby("project_id")["user_id"].apply(lambda x: len(set(x))),
        col_name="nb_users",
        idx_name="user_id"
    )


    return nb_wkrs_per_prj

def time_exp_project(df_decs):
    """
    Computes:
      1) Project lifetime (in days): (max_declared_at - min_declared_at).
      2) Users' "experience" counts per project (based on the sequence of first declarations).
      3) Mean deviation of users' experience per project.
    """

    # 1) Compute project lifetime in days
    lftm_per_prj = (
        df_decs
        .groupby("project_id")
        .apply(lambda x: (x["declared_at"].max() - x["declared_at"].min()).days)
        .reset_index()
        .rename({0: "lftm"}, axis=1)
    )

    # 2) Determine user "experience" per project
    #    - Get each user's earliest declaration date for each project.
    min_day_usr_prj = df_decs.groupby(["user_id", "project_id"])["declared_at"].min().sort_values()

    #    - Compute an incremental count ("exp") based on the chronological order 
    #      of each user's first time joining a new project.
    cnt = (
        min_day_usr_prj
        .reset_index()
        .groupby("user_id")
        .apply(
            lambda x: (
                # Compare each date to the previous one; if equal, mark as repeated.
                # Otherwise, increment a counter.
                np.ones(len(x)) - (x["declared_at"].shift().eq(x["declared_at"])).astype(int)
            ).cumsum() - 1
        )
    )

    #    - Build a DataFrame with (user_id, project_id, experience)
    exp_prj_usr = pd.concat(
        [
            pd.DataFrame(min_day_usr_prj.loc[u])
            .assign(user_id=u, exp=cnt.loc[u].values)
            for u in df_decs.user_id.drop_duplicates()
        ],
        axis=0
    ).reset_index()[["user_id", "project_id", "exp"]]

    # 3) Compute mean of users' experience by project
    mean_exp = (
        exp_prj_usr
        .groupby("project_id")["exp"]
        .mean()
        .reset_index()
        .rename({"exp": "mean_exp"}, axis=1)
    )

    return mean_exp, lftm_per_prj


def diversity_project(df_decs, df_users, df_projects):
    """
    Calculates: A 'diversity' metric for past projects within teams.
    """

    # 1) Prepare a DataFrame of projects and their first/last declaration dates per user/team
    comp_proj = (
        df_decs.merge(
            pd.merge(
                df_decs.groupby("project_id")["declared_at"].min().reset_index().rename({"declared_at": "first"}, axis=1),
                df_decs.groupby("project_id")["declared_at"].max().reset_index().rename({"declared_at": "last"}, axis=1),
                on="project_id"
            )
        )
        .merge(df_users[["user_id", "team_id"]])
        .merge(df_projects[df_projects.is_active == False][["project_id"]])  # Consider only inactive projects
    )[["project_id", "user_id", "team_id", "first", "last"]].drop_duplicates()

    # 2) Build a dict of past projects (per team) based on date comparisons
    past_projects_team = {}
    for team in tqdmn(comp_proj.team_id.drop_duplicates()):
        sel_comp_proj = comp_proj[comp_proj.team_id == team]
        first_dates = sel_comp_proj['first'].values[:, None]  # Column vector
        last_dates  = sel_comp_proj['last'].values            # Row vector
        comparison_matrix = first_dates > last_dates          # True if current project's first > other project's last
        project_ids = sel_comp_proj['project_id'].values
        past_projects_team[team] = {
            project_id: project_ids[comparison_matrix[i]]
            for i, project_id in enumerate(project_ids)
        }

    # 3) Similarly, build a dict of past projects for each user
    past_projects_user = {}
    for user in tqdmn(comp_proj.user_id.drop_duplicates()):
        sel_comp_proj = comp_proj[comp_proj.user_id == user]
        first_dates = sel_comp_proj['first'].values[:, None] 
        last_dates  = sel_comp_proj['last'].values
        comparison_matrix = first_dates > last_dates
        project_ids = sel_comp_proj['project_id'].values
        past_projects_user[user] = {
            project_id: project_ids[comparison_matrix[i]]
            for i, project_id in enumerate(project_ids)
        }

    # 4) Create lookup structures for user-team and project-user relationships
    prj2user  = (
        pd.DataFrame([(proj, user) for user, dic in past_projects_user.items() for proj in dic.keys()],
                     columns=["project_id", "user_id"])
        .groupby("project_id")["user_id"]
        .apply(lambda x: list(set(x)))
    )

    # 5) Compute 'diversity' for each project based on how many past projects 
    #    each user was involved in and how distinct these past projects are overall.
    diversity = pd.DataFrame([
        (
            proj,
            len(
                set([k for user in prj2user[proj] for k in past_projects_user[user][proj]])
            ) / np.float64(
                sum([len(set(past_projects_user[user][proj])) for user in prj2user[proj]]) + 0.0
            )
        )
        for team, dic in past_projects_team.items()
        for proj, past_projects in dic.items()
    ], columns=["project_id", "diversity"]).dropna()


    return diversity

def results_summary_to_dataframe(results):
    """
    Convert a statsmodels results table into a Pandas DataFrame
    with coefficients, p-values, and confidence intervals.
    """
    # Extract relevant statistics from the statsmodels results
    pvals = results.pvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    # Create a DataFrame with p-values, coefficients, and lower/upper CI
    results_df = pd.DataFrame({
        "p-val": pvals,
        "r": coeff,
        "ll": conf_lower,
        "hl": conf_higher
    })

    # Reorder columns for clarity
    results_df = results_df[["r", "p-val", "ll", "hl"]]
    
    return results_df