from copy import deepcopy
from sys import argv

from seminar.src.old_version.gp_symreg import *

"""y
Description:
    Do a full run (Train and Test) for both algorithms 

Usage:
    ARGV[1]: Target Variable (y1 | y2)
    ARGV[2]: Error Metric (absolute | squared) 
"""


def run(target_var: str, err_metric: str):
    """Train, Test and log results_50gens_500pop for <target_var> using mean <err_metric> error for minimizing fitness function"""
    # get database and split into two pd.DataFrames
    TRAINING_DATASET, TESTING_DATASET = get_datasets(training_split=0.5)

    # primitive set
    PRIMITIVE_SET = get_primitive_set()

    # deap toolbox
    TOOLBOX = get_toolbox(
        primitive_set=PRIMITIVE_SET,
    )

    # deap multi statistics
    MULTI_STATISTICS = get_multi_statistics()

    # gp configuration
    EA_CONFIG = EaConfig(
        population_size=50, #500
        num_generations=10, #50
        mutation_rate=0.05,
        crossover_rate=0.2,
        multi_statistics=MULTI_STATISTICS,
        target_var=target_var,
        err_metric=err_metric,
    )

    # train model for y1 using tournament selection
    training_results_tournament = train_tournament(
        toolbox=deepcopy(TOOLBOX),
        ea_config=EA_CONFIG,
        training_dataset=TRAINING_DATASET,
        testing_dataset=TESTING_DATASET,
        pset=deepcopy(PRIMITIVE_SET)
    )

    # # test model on unknown dataset
    # testing_results_tournament = test_model(
    #     training_results_tournament,
    #     EA_CONFIG,
    #     TESTING_DATASET,
    #     PRIMITIVE_SET,
    # )

    # # log single run results_50gens_500pop
    # log_single_run(
    #     ea_results=training_results_tournament,
    #     testing_results=testing_results_tournament,
    #     selection_method="tournament"
    # )

    # train model for y1 using epsilon lexicase selection
    training_results_epsilon_lexicase = train_epsilon_lexicase(
        toolbox=deepcopy(TOOLBOX),
        ea_config=EA_CONFIG,
        training_dataset=TRAINING_DATASET,
        testing_dataset=TESTING_DATASET,
        pset=deepcopy(PRIMITIVE_SET)
    )

    # # test model on unknown dataset
    # testing_results_epsilon_lexicase = test_model(
    #     training_results_epsilon_lexicase,
    #     EA_CONFIG,
    #     TESTING_DATASET,
    #     PRIMITIVE_SET,
    # )

    # # log single run results_50gens_500pop
    # log_single_run(
    #     ea_results=training_results_epsilon_lexicase,
    #     testing_results=testing_results_epsilon_lexicase,
    #     selection_method="epsilon-lexicase"
    # )

    # # append results_50gens_500pop to master record
    # full_results = FullResults(
    #     algorithm_1="tournament_selection",
    #     training_results_1=training_results_tournament,
    #     testing_results_1=testing_results_tournament,
    #     algorithm_2="epsilon_lexicase_selection",
    #     training_results_2=training_results_epsilon_lexicase,
    #     testing_results_2=testing_results_epsilon_lexicase
    # )

    # log_full_results(full_results, f"../results_50gens_500pop/master_log.csv")


if __name__ == "__main__":
    run(argv[1], argv[2])
