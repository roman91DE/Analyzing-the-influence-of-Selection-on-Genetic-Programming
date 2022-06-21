#!/usr/bin/env python
# coding: utf-8

# # Statistics
# 
# This notebook needs to be executed to create all the plots and tables of statistical tests that are referenced in the main paper.Rmd file. Output is created from the csv files located at ../results/single_run
# 
# 
# The initial results can be reproduced by running the shell script <run.zsh>
# 
# All results of this notebook are saved in ../docs/rmd/plots and ../docs/rmd/tables

# In[185]:


from scipy.stats import mannwhitneyu, normaltest
import pandas as pd
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from typing import Tuple, List
from statistics import mean
from dataclasses import dataclass
import csv
import os

get_ipython().run_line_magic('cd', '~/github/geneticProgramming/seminar/src')


# In[186]:


# plotting setup

get_ipython().run_line_magic('matplotlib', 'inline')

FIGSIZE_INCHES =  14, 9
FIGSIZE_INCHES_LARGE =  16, 10
TITLE_FONT_SIZE = 17
TITLE_FONT_SIZE_LARGE = 20


# In[187]:


# set output paths

PATH = "../docs/rmd/"
TABLE_PATH = f"{PATH}/tables"
PLOT_PATH =  f"{PATH}/plots"

jobs = (
    'os.makedirs(f"{TABLE_PATH}/csv")',
    'os.makedirs(f"{TABLE_PATH}/md")',
    'os.makedirs(f"{PLOT_PATH}")'
)

for job in jobs:
    try:
        exec(job)
    except FileExistsError:
        pass

# set input paths

DIR_TOURNAMENT = "../results/single_run/e_lexicase"
DIR_ELEXICASE = "../results/single_run/tournament"


tournament_files = os.listdir(DIR_TOURNAMENT)
elexicase_files = os.listdir(DIR_ELEXICASE)

if not len(tournament_files) == len(elexicase_files):
    print("Warning - Unequal number of records!\nVariable <TOTAL_RUNS> not set")
    TOTAL_RUNS = None
    
else:
    TOTAL_RUNS = len(tournament_files)
    print(f"Total number of Runs: {TOTAL_RUNS}")


# In[188]:


def tsv_to_df(filepath: str) -> pd.DataFrame:
    """Return the results for ../results/single_run/<algorithm><id>.tsv as a pd.DataFrame"""

    df = pd.read_csv(filepath_or_buffer=filepath, sep="\t", index_col=False, skipinitialspace=True)
    
    df.columns = df.columns.str.strip()
    
    rename_dict = {
        "avg" : "mean_training_error",
        "std" : "std_training_error",
        "min" : "min_training_error",
        "max" : "max_training_error",
        "elite_testing_mse" : "testing_error",
        "elite_testing_err_std" : "std_testing_error"
        
    }
        
    return df.rename(columns=rename_dict) 
        


# read and store all log files into dataframes
tournament_logs = []
elexicase_logs = []

for a, b in zip(tournament_files, elexicase_files):
    tournament_logs.append(
        tsv_to_df(f"{DIR_TOURNAMENT}/{a}")
    )
    elexicase_logs.append(
        tsv_to_df(f"{DIR_ELEXICASE}/{b}")
    )
    


# # Descriptive Statistics

# In[189]:


# print all individual logs

for idx, (a, b) in enumerate(zip(tournament_logs, elexicase_logs)):
    print(f"{idx+1}.th Run:\nTournament-Selection:\n{a}\nE-Lexicase-Selection:\n{b}\n--------------\n")


# In[190]:


def to_master_record(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    
    """
    Summarize and return the results from each individual dataframe into a master record
    """
    
    headers = dfs[0].columns.values.tolist()
    ngens = len(dfs[0]["gen"])
    
    master = pd.DataFrame(0, index=np.arange(ngens), columns=headers)
    
    def mean_stddev(std_devs: List[float]) -> float:
        """returns the mean for a list of std_deviations """
        agg = 0.0
        for std_dev in std_devs:
            agg += std_dev ** 2
        return sqrt(agg / len(std_devs))
    
    for header in headers:
                
        for gen in range(ngens):
            
            vals = []
            
            for df in dfs:
                vals.append(
                    float(df[header].iloc[gen])
                )

            if not "std" in header:
                master.loc[gen,header] = mean(vals)
            
            else:
                master.loc[gen,header] = mean_stddev(vals)
                
    return master



master_tournmament = to_master_record(tournament_logs)
master_elexicase = to_master_record(elexicase_logs)
    


# In[191]:


master_tournmament.to_csv(path_or_buf=f"{TABLE_PATH}/csv/master_tournament.csv")
master_tournmament.to_markdown(buf=f"{TABLE_PATH}/md/master_tournament.md")

master_tournmament


# In[192]:


master_tournmament.describe().to_csv(path_or_buf=f"{TABLE_PATH}/csv/master_tournament_descriptive.csv")
master_tournmament.describe().to_markdown(buf=f"{TABLE_PATH}/md/master_tournament_descriptive.md")

master_tournmament.describe()


# In[193]:


master_elexicase.to_csv(path_or_buf=f"{TABLE_PATH}/csv/master_elexicase.csv")
master_elexicase.to_markdown(buf=f"{TABLE_PATH}/md/master_elexicase.md")

master_elexicase


# In[194]:


master_elexicase.describe().to_csv(path_or_buf=f"{TABLE_PATH}/csv/master_elexicase_descriptive.csv")
master_elexicase.describe().to_markdown(buf=f"{TABLE_PATH}/md/master_elexicase_descriptive.md")

master_elexicase.describe()


# In[195]:


def aggregate_cells(dfs: List[pd.DataFrame], header, row) -> List[float]:
    
    vals = []

    for df in dfs:
        
        vals.append(
            df[header].iloc[row]
        )
    
    return vals

LAST_ROW = len(tournament_logs[0]) - 1

# aggregate training errors for elite models in last generation
tournament_elite_training_errors = aggregate_cells(tournament_logs, "min_training_error", LAST_ROW)
elexicase_elite_training_errors = aggregate_cells(elexicase_logs, "min_training_error", LAST_ROW)

# aggregate elite model performance on testing data
tournament_elite_testing_errors = aggregate_cells(tournament_logs, "testing_error", LAST_ROW)
elexicase_elite_testing_errors = aggregate_cells(elexicase_logs, "testing_error", LAST_ROW)


# aggregate size values

tournament_elite_size = aggregate_cells(tournament_logs, "elite_size", LAST_ROW)
elexicase_elite_size = aggregate_cells(elexicase_logs, "elite_size", LAST_ROW)

# aggregate elite model performance on testing data
tournament_avg_size = aggregate_cells(tournament_logs, "avg_size", LAST_ROW)
elexicase_avg_size = aggregate_cells(elexicase_logs, "avg_size", LAST_ROW)


# In[196]:


# test if samples are normal distributed at alpha=5%, results are written to ../docs/rmd/tables/normal_dist_test.csv"

def is_normal_distr(vals: List[float], name: str, alpha:float=0.05) -> str:  
    """
    Null Hypothesis: Sample comes from a normal distribution, 
    returns results as csv string:
        <sample,statistic,p-value,alpha,normal_distributed>
    
    """
    statistic, pval = normaltest(vals)
    return f"{name},{statistic},{pval},{alpha},{pval >= alpha}\n"
    

csv_str = (
    "sample,statistic,p-value,alpha,normal_distributed\n" +
    is_normal_distr(tournament_elite_training_errors, "Tournament - Training Errors") +
    is_normal_distr(elexicase_elite_training_errors, "E-Lexicase - Training Errors") +
    is_normal_distr(tournament_elite_testing_errors, "Tournament - Testing Errors") +
    is_normal_distr(elexicase_elite_testing_errors, "E-Lexicase - Testing Errors")
)

print(csv_str)

        
with open(f"../docs/rmd/tables/csv/normal_dist_test.csv", "w") as fstr:
    fstr.write(csv_str)
    


# In[197]:


def test_mannwhitneyu(sample_a: List[float], sample_b: List[float], alpha:float=0.05) -> Tuple[float,float]:
    """
    performs a mann whitney u ranksum test for sample_a and sample_b, 
    returns the results as csv string
        <test statistic and p-value>
    """
    statistic, pval = mannwhitneyu(x = sample_a,y = sample_b)
    print(f"Statistic: {statistic}\nPVal: {pval}\nPVal < ALPHA: {pval < alpha}")

    if pval > alpha:
        print(f"Results supports H0 for alpha={alpha}\n H0: The distribution underlying sample_a is the same as the distribution underlying sample_b")

    else:
        print(f"H0 can be rejected for alpha={alpha}\nThe distribution underlying sample_a is NOT the same as the distribution underlying sample_b")
    
    return statistic, pval


@dataclass
class Sample:
    vals: List[float]
    name: str

        
def mark(pval:float) -> str:
        """
        mark pvalues for statistical significance:
            alpha:
                0.1   : *
                0.05  : **
                0.025 : ***
        """
        s = str(pval)
        
        if pval < 0.1:
            s += '*'
        if pval < 0.05:
            s += '*'
        if pval < 0.025:
            s += '*'
            
        return s        


def mwu_csv_matrix(
    samples: List[Sample],
    alpha:float=0.05
) -> str:
    """
    returns the results of mwu test as a csv matrix

    """
    
    # x 0 1 2 3
    # 0
    # 1
    # 2
    # 3
    
    
    names = []
    
    for sample in samples:
        names.append(sample.name)
    
    matrix = [[None for _ in range(len(samples))] for _ in range(len(samples))]
    
    for ix, xsample in enumerate(samples):
        for iy, ysample in enumerate(samples):
            _, p = mannwhitneyu(x=xsample.vals, y=ysample.vals)
            matrix[ix][iy] = p
            
    csv_str = "{0},{1},{2},{3}\n".format(*[name for name in names])
    
    for row,name in zip(matrix, names):
        csv_str += "{},{},{},{},{}\n".format(name, *row)
        
    return csv_str


# In[198]:


# MWU - Error

samples = [
    Sample(tournament_elite_training_errors, "tournament_training_errors"),
    Sample(tournament_elite_testing_errors, "tournament_testing_errors"),
    Sample(elexicase_elite_training_errors, "elexicase_training_errors"),
    Sample(elexicase_elite_testing_errors, "elexicase_testing_errors") 
]


csv = mwu_csv_matrix(samples)

with open("../docs/rmd/tables/csv/mwu_matrix_error.csv", "w") as f:
    f.write(csv)



pd.read_csv(open("../docs/rmd/tables/csv/mwu_matrix_error.csv"))


# In[199]:


# MWU - Size

# MWU - Error

samples_size = [
    Sample(tournament_elite_size, "tournament_elite_size"),
    Sample(elexicase_elite_size, "elexicase_elite_size"),
    Sample(tournament_avg_size, "tournament_avg_size"),
    Sample(elexicase_avg_size, "elexicase_avg_size") 
]


csv = mwu_csv_matrix(samples_size)

with open("../docs/rmd/tables/csv/mwu_matrix_size.csv", "w") as f:
    f.write(csv)



pd.read_csv(open("../docs/rmd/tables/csv/mwu_matrix_size.csv"))


# In[200]:


def save_as_boxplots(
    sample_a: List[float],
    sample_b: List[float],
    title: str,
    a_label:str, b_label: str, filename: str) -> None:

    PATH = f"../docs/rmd/plots/{filename}.png"

    fig, ax = plt.subplots()
    fig.set_size_inches(*FIGSIZE_INCHES)
    
    plt.grid(visible=True, axis='both')

    ax.boxplot(
        x = [sample_a, sample_b],
        labels=[a_label, b_label]
    )

    ax.set_title(title, fontsize=TITLE_FONT_SIZE_LARGE)
    ax.set_ylabel("MSE")
    plt.savefig(PATH)
    plt.show()
    
def save_as_boxplots_all(
    sample_a: List[float],
    sample_b: List[float],
    sample_c: List[float],
    sample_d: List[float],
    title: str,
    a_label:str, b_label: str, c_label:str,d_label: str,
    filename: str
) -> None:

    PATH = f"../docs/rmd/plots/{filename}.png"

    fig, ax = plt.subplots()
    fig.set_size_inches(*FIGSIZE_INCHES_LARGE)
    
    plt.grid(visible=True, axis='both')

    ax.boxplot(
        x = [sample_a, sample_b, sample_c, sample_d],
        labels=[a_label, b_label, c_label, d_label]
    )
    
    ax.set_title(title, fontsize=TITLE_FONT_SIZE_LARGE)
    ax.set_ylabel("MSE")
    plt.savefig(PATH)
    plt.show()


# In[201]:


test_mannwhitneyu(tournament_elite_training_errors, elexicase_elite_training_errors)

save_as_boxplots(
    tournament_elite_training_errors,
    elexicase_elite_training_errors,
    "Training Errors Distribution",
    "Tournament-Selection",
    "Epsilon-Lexicase-Selection",
    "mean_training_errors_boxplot"
)


# In[202]:


test_mannwhitneyu(tournament_elite_testing_errors, elexicase_elite_testing_errors)

save_as_boxplots(
    tournament_elite_testing_errors,
    elexicase_elite_testing_errors,
    "Testing Errors Distribution",
    "Tournament-Selection",
    "Epsilon-Lexicase-Selection",
    "mean_testing_errors_boxplot"
)


# In[203]:


save_as_boxplots_all(
    sample_a=tournament_elite_testing_errors,
    sample_b=tournament_elite_training_errors,
    sample_c=elexicase_elite_testing_errors,
    sample_d=elexicase_elite_training_errors,
    title="Testing Errors Distribution",
    a_label="Tournament_Testing",
    b_label="Tournament_Training",
    c_label="E_Lexicase_Testing",
    d_label="E_Lexicase_Training",
    filename="mean_error_boxplot_all"
)


# In[ ]:





# In[204]:


# testing error gap

def evolutionary_plot(
    master_record: pd.DataFrame, 
    header_1: str, 
    header_2: str, 
    algorithm_name: str, 
    filename: str, 
    suptitle: str,
    y_scale: Tuple[int, int]=(0,100)
):
    
    
    PATH = f"../docs/rmd/plots/{filename}.png"
    
    X = np.arange(
        min(master_record["gen"]),
        max(master_record["gen"] +1)
    )
    
    fig, ax = plt.subplots()
    fig.set_size_inches(*FIGSIZE_INCHES)
    
    
    ax.plot(X, master_record[header_1], label=header_1)
    ax.plot(X, master_record[header_2], label=header_2)
    
    ax.set_title(f"{suptitle} - {algorithm_name}", fontsize=TITLE_FONT_SIZE_LARGE)
    ax.set_ylim(*y_scale)
    
    ax.set_xlabel("generations")
    ax.set_ylabel("MSE")
    ax.legend()
    
    plt.grid(visible=True, axis='both')
    plt.savefig(PATH)
    plt.show()
    
    
# testing error gap

def evolutionary_masterplot(
    master_record_1: pd.DataFrame,
    master_record_2: pd.DataFrame,
    total_runs: int,
    header_1: str,
    header_2: str,
    algorithm_1_name: str,
    algorithm_2_name: str,
    filename: str,
    suptitle: str,
    y_label: str,
    y_scale: Tuple[int, int]=(0,100),
):
    
    
    PATH = f"../docs/rmd/plots/{filename}.png"
    
    X = np.arange(
        min(master_record_1["gen"]),
        max(master_record_2["gen"] +1)
    )
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(*FIGSIZE_INCHES_LARGE)
    
    
    ax1.plot(X, master_record_1[header_1], label=header_1)
    ax1.plot(X, master_record_1[header_2], label=header_2)
    ax1.grid(True)
    
    ax2.plot(X, master_record_2[header_1], label=header_1)
    ax2.plot(X, master_record_2[header_2], label=header_2)
    ax2.grid(True)
    
    ax1.set_title(algorithm_1_name, fontsize=TITLE_FONT_SIZE)
    ax2.set_title(algorithm_2_name, fontsize=TITLE_FONT_SIZE)
    
    ax1.set_ylim(*y_scale)
    ax2.set_ylim(*y_scale)
    
    ax1.set_xlabel("generations")
    ax1.set_ylabel(y_label)
    
    ax2.set_xlabel("generations")
    ax2.set_ylabel("MSE")
    
    ax1.legend()
    ax2.legend()
    
    plt.suptitle(f"{suptitle} for {total_runs} total Runs", fontsize=TITLE_FONT_SIZE_LARGE)
    
    plt.savefig(PATH)
    plt.show()
    
    
# testing error gap

def evolutionary_combined_masterplot(
    master_record_1: pd.DataFrame,
    master_record_2: pd.DataFrame,
    total_runs: int,
    header_1: str,
    header_2: str,
    algorithm_1_name: str,
    algorithm_2_name: str,
    filename: str,
    suptitle: str,
    y_label: str,
    y_scale: Tuple[int, int]=(0,100),
):
    
    
    PATH = f"../docs/rmd/plots/{filename}.png"
    
    X = np.arange(
        min(master_record_1["gen"]),
        max(master_record_2["gen"] +1)
    )
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(*FIGSIZE_INCHES_LARGE)
    
    
    ax.plot(X, master_record_1[header_1], "b" , label=f"{algorithm_1_name}_{header_1}")
    ax.plot(X, master_record_1[header_2], "g" , label=f"{algorithm_1_name}_{header_2}")
    ax.plot(X, master_record_2[header_1], "y", label=f"{algorithm_2_name}_{header_1}")
    ax.plot(X, master_record_2[header_2], "r",label=f"{algorithm_2_name}_{header_2}")
    
    ax.grid(True)

    
    ax.set_ylim(*y_scale)
    ax.set_ylim(*y_scale)
    
    ax.set_ymargin(1.5)
    
    ax.set_xlabel("generations")
    ax.set_ylabel(y_label)

    ax.legend()
    
    plt.suptitle(f"{suptitle} for {total_runs} total Runs", fontsize=TITLE_FONT_SIZE)
    
    plt.savefig(PATH)
    plt.show()
    
    


# # Mean Error - Plots

# In[205]:


evolutionary_plot(
    master_record=master_tournmament,
    header_1="min_training_error",
    header_2="testing_error",
    algorithm_name="Tournament Selection",
    filename="tournament_evolution",
    suptitle="Mean Error"
)
evolutionary_plot(
    master_record=master_elexicase,
    header_1="min_training_error",
    header_2="testing_error",
    algorithm_name="Epsilon-Lexicase Selection",
    filename="elexicase_evolution",
    suptitle="Mean Error"
)


# In[206]:


evolutionary_masterplot(
    master_record_1=master_tournmament,
    master_record_2=master_elexicase,
    total_runs=TOTAL_RUNS,
    header_1="min_training_error",
    header_2="testing_error",
    algorithm_1_name="Tournament Selection",
    algorithm_2_name="Epsilon-Lexicase Selection",
    filename="mean_error_subplotted",
    suptitle="Mean Error",
    y_label="MSE"
)


# In[207]:



evolutionary_combined_masterplot(
master_record_1=master_tournmament,
master_record_2=master_elexicase,
total_runs=TOTAL_RUNS,
header_1="min_training_error",
header_2="testing_error",
algorithm_1_name="Tournament Selection",
algorithm_2_name="Epsilon-Lexicase Selection",
filename="mean_error_combined",
suptitle="Mean Error",
y_label="MSE"
)


# # Mean Size Plots

# In[208]:


evolutionary_masterplot(
    master_record_1=master_tournmament,
    master_record_2=master_elexicase,
    total_runs=TOTAL_RUNS,
    header_1="avg_size",
    header_2="elite_size",
    algorithm_1_name="Tournament Selection",
    algorithm_2_name="Epsilon-Lexicase Selection",
    filename="size_subplotted",
    suptitle="Mean Size",
    y_label="size"
)


# In[209]:


evolutionary_combined_masterplot(
    master_record_1=master_tournmament,
    master_record_2=master_elexicase,
    total_runs=TOTAL_RUNS,
    header_1="avg_size",
    header_2="elite_size",
    algorithm_1_name="Tournament Selection",
    algorithm_2_name="Epsilon-Lexicase Selection",
    filename="size_combined",
    suptitle="Mean Size",
    y_label="size"
)


# In[210]:


# add, commit and push to github remote repo
get_ipython().system(' git add * && git commit -m "working in jupyter notebook" && git push')

