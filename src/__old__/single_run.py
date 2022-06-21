from copy import deepcopy
from sys import argv

from seminar.src.old_version.gp_symreg import *


def run(target_var: str, err_metric: str):
    """Train, Test and log results_50gens_500pop for <target_var> using mean <err_metric> error for minimizing fitness function"""
    # get database and split into two pd.DataFrames
    TRAINING_DATASET, TESTING_DATASET = get_datasets(training_split=0.65)

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
        population_size=500,
        num_generations=40,
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
    )

    # test model on unknown dataset
    testing_results_tournament = test_model(
        training_results_tournament,
        EA_CONFIG,
        TESTING_DATASET,
        PRIMITIVE_SET,
    )

    # train model for y1 using epsilon lexicase selection
    training_results_epsilon_lexicase = train_epsilon_lexicase(
        toolbox=deepcopy(TOOLBOX),
        ea_config=EA_CONFIG,
        training_dataset=TRAINING_DATASET,
    )

    # test model on unknown dataset
    testing_results_epsilon_lexicase = test_model(
        training_results_epsilon_lexicase,
        EA_CONFIG,
        TESTING_DATASET,
        PRIMITIVE_SET,
    )

    full_results = FullResults(
        algorithm_1="Tournament Selection",
        training_results_1=training_results_tournament,
        testing_results_1=testing_results_tournament,
        algorithm_2="Epsilon-Lexicase Selection",
        training_results_2=training_results_epsilon_lexicase,
        testing_results_2=testing_results_tournament
    )

    # log results_50gens_500pop
    directory = log_single_run(
        ea_results=training_results_tournament,
        testing_results=testing_results_tournament,
        ea_config=EA_CONFIG,
        filename=f"tournament_{target_var}_{err_metric}",
    )

    log_single_run(
        ea_results=training_results_epsilon_lexicase,
        testing_results=testing_results_epsilon_lexicase,
        ea_config=EA_CONFIG,
        filename=f"epsilon_lexicase_{target_var}_{err_metric}",
        directory=directory,
    )

    # plot results_50gens_500pop
    PLOTS = (
        ("fitness", "min"),
        ("fitness", "avg"),
        ("fitness", "std"),
        ("size", "avg"),
    )

    for metric, stats in PLOTS:
        separate_plots_evolution(
            attribute=metric,
            statistic_function=stats,
            ea_results_1=training_results_tournament,
            ea_results_2=training_results_epsilon_lexicase,
            algorithm_1="tournament",
            algorithm_2="epsilon-lexicase",
            directory=directory,
        )

        combined_plots_evolution(
            attribute=metric,
            statistic_function=stats,
            ea_results_1=training_results_tournament,
            ea_results_2=training_results_epsilon_lexicase,
            algorithm_1="tournament",
            algorithm_2="epsilon-lexicase",
            directory=directory,
        )


if __name__ == "__main__":
    run(argv[1], argv[2])
