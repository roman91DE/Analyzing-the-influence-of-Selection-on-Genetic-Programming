from scipy.stats import mannwhitneyu
import pandas as pd


def main():

    # load results_50gens_500pop into pd.DataFrame
    data_path = "../results_50gens_500pop/multiple_runs.csv"
    df = pd.read_csv(
        filepath_or_buffer=data_path,
        sep=",",
        header=0,
    )

    print(df.head())


if __name__ == "__main__":
    main()