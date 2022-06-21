import math
import operator
import os
from dataclasses import dataclass
from random import random, randint
from statistics import pstdev
from sys import stderr
from typing import Any, Tuple, Callable, List

import networkx as nx
import numpy as np
import pandas as pd
from deap import gp, tools, creator, base, algorithms
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as skl_train_test_split


def get_datasets(training_split: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download the original database, splits and returns two randomly sampled datasets for training and testing

    Usage:
        0.0 < <training_split> < 1.0
        size training dataset: N *       <training_split>
        size testing dataset:  N *  (1 - <training_split>)
    """

    assert 0.0 < training_split < 1.0

    FILENAME = "ENB2012_data.xlsx"
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/"
    DESTINATION = "../data"

    def __download() -> None:
        if not os.path.exists(f"{DESTINATION}/{FILENAME}"):
            os.system(f"wget {URL}{FILENAME} -P {DESTINATION}")

    def __split(
            __training_split: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_excel(f"{DESTINATION}/{FILENAME}")
        return skl_train_test_split(
            df,
            train_size=__training_split,
            test_size=(1.0 - __training_split),
        )

    __download()
    return __split(training_split)


def plot_expression_tree(expr_tree, title: str, filepath: str) -> None:
    """plots an expression tree and saves it as ../plots/<filename>.png"""
    nodes, edges, labels = gp.graph(expr_tree)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")

    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)

    plt.title(title)
    plt.savefig(filepath)


def get_primitive_set() -> gp.PrimitiveSet:
    """returns a basic deap.gp.PrimitiveSet object for symbolic regression without registering methods for selection
    and fitness evaluation """
    UVS = {
        "ARG0": "X1",
        "ARG1": "X2",
        "ARG2": "X3",
        "ARG3": "X4",
        "ARG4": "X5",
        "ARG5": "X6",
        "ARG6": "X7",
        "ARG7": "X8",
    }

    class ProtectedOperators:
        """Wrapper class for protected versions of operators"""

        @staticmethod
        def div(lhs: float, rhs: float) -> float:
            """
            Koza Style implementation of division
            [@Koza2005]
            """
            if rhs == 0:
                return 1
            return lhs / rhs

        @staticmethod
        def log(x: float) -> float:
            """
            Koza Style implementation of natural logarithm
            [@Koza2005]
            """
            if x == 0:
                return 0
            return math.log(abs(x))

        @staticmethod
        def sqrt(x: float) -> float:
            """
            Koza Style implementation of square root
            [@Koza2005]
            """
            return math.sqrt(abs(x))

        @staticmethod
        def pow(num: float, power: float) -> float:
            """
            Adjusted Implementation of power operator
            [@fsets_generalisation]
            """
            if (num != 0) or (num == power == 0):
                return abs(num) ** power
            return 0

    # register the Primitive Set
    pset = gp.PrimitiveSet("MAIN", arity=len(UVS))

    # rename ARGS to match the dataset
    for arg, des in UVS.items():
        pset.renameArguments(arg=des)

    operators = (
        (operator.add, 2),
        (operator.sub, 2),
        (operator.mul, 2),
        (operator.neg, 1),
        (math.sin, 1),
        (math.cos, 1),
        (ProtectedOperators.div, 2),
        (ProtectedOperators.log, 1),
        (ProtectedOperators.sqrt, 1),
        # (ProtectedOperators.pow, 2),  -> Overflow Errors!
    )

    for (func, arity) in operators:
        pset.addPrimitive(func, arity)

    pset.addEphemeralConstant("random_int", lambda: randint(-10, 10))
    pset.addEphemeralConstant("random_float", lambda: random())

    return pset


def get_toolbox(
        primitive_set: gp.PrimitiveSet,
) -> base.Toolbox:
    """returns a basic deap.base.Toolbox object for symbolic regression"""

    # create custom types
    # min fitness object: objective: minimize mse/mae for y1^/y2^
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # individuals program: basic tree with min fitness
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # TODO: research optimal configuration from literature

    # basic gp setup
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=primitive_set, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=primitive_set)

    # genetic operators
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitive_set)

    # bloat control
    toolbox.decorate(
        "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
    )
    toolbox.decorate(
        "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
    )

    return toolbox


def evaluate_single_case(
        func: Callable, case: pd.Series, target_var: str, err_metric: str
) -> float:
    """
    Evaluates an individual, compiled program for a single fitness case,
    computes and returns error for prediction and outcome for target_var and model prediction

    Options:

        target_var:
            "y1" (heating load)
            "y2" (cooling load)

        err_metric:
            "squared" (error)
            "absolute" (error)

    """
    assert (target_var.lower() == "y1") or (target_var.lower() == "y2")

    # compute individual with case variables
    prediction = func(*case[0:8:].values)

    # optimal value:
    if target_var.lower() == "y1":
        value = case.values[8]
    elif target_var == "y2":
        value = case.values[9]
    else:
        print("Error occurred during call to evaluate_single_case()!", file=stderr)
        exit()

    # compute and return error as defined by err_metric
    if err_metric.lower() == "squared":
        return (prediction - value) ** 2
    elif err_metric.lower() == "absolute":
        return abs(prediction - value)
    else:
        print("Error occurred during call to evaluate_single_case()!", file=stderr)
        exit()


def evaluate_all_cases(
        individual: "creator.Individual",
        toolbox: base.Toolbox,
        dataset: pd.core.frame.DataFrame,
        target_var: str,
        err_metric: str,
) -> tuple[float]:
    """
    Evaluates an individual program for all fitness cases inside the dataset,
    computes and returns the mean for err_metric of prediction and target_var
    """
    # compile the tree expression into a callable function
    compiled_individual = toolbox.compile(expr=individual)

    n = len(dataset)
    error_aggregate = 0.0

    # iterate through all fitness cases and aggregate absolute errors
    for _, fitness_case in dataset.iterrows():
        error_aggregate += evaluate_single_case(
            func=compiled_individual,
            case=fitness_case,
            target_var=target_var,
            err_metric=err_metric,
        )

    # compute and return mean error
    mean_error = error_aggregate / n
    return mean_error,


def get_multi_statistics() -> "tools.MultiStatistics":
    """returns a deap.tools.MultiStatics object"""
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    # stats_testing_err = tools.Statistics()
    # mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, testing_error=stats_testing_err)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    return mstats


@dataclass
class EaConfig:
    """wrapper for basic EA configurations"""

    population_size: int
    num_generations: int
    mutation_rate: float
    crossover_rate: float
    multi_statistics: "tools.MultiStatistics"
    target_var: str
    err_metric: str

    def log_string(self):
        return f"""population_size: {self.population_size}
num_generations: {self.num_generations}
mutation_rate: {self.mutation_rate}
crossover_rate: {self.crossover_rate}
target_variable: {self.target_var}
error_metric: {self.err_metric}
EOF
"""


@dataclass
class TrainingResults:
    """wrapper for return values from deap.algorithms.eaSimple()"""

    population: Any
    logbook: tools.Logbook
    hall_of_fame: tools.HallOfFame
    mean_error_testing: float = -1.00
    std_dev_testing: float = -1.00


@dataclass
class TestingResults:
    """wrapper for testing results_50gens_500pop"""
    mean_error: float
    std_dev: float


@dataclass
class FullResults:
    """Wrapper for all results_50gens_500pop from testing and training for both algorithms"""
    algorithm_1: str
    training_results_1: TrainingResults
    testing_results_1: TestingResults

    algorithm_2: str
    training_results_2: TrainingResults
    testing_results_2: TestingResults

    # chapters[attribute].select("min")

    def get_csv_str(self):
        """
        returns a string in .csv format for logging:
        Format:
            alg1_training_min_fitness | alg1_testing_min_fitness | alg2_training_min_fitness | alg2_testing_min_fitness"""
        return f"""{min(self.training_results_1.logbook.chapters["fitness"].select("min"))},{self.testing_results_1.mean_error},{min(self.training_results_2.logbook.chapters["fitness"].select("min"))},{self.testing_results_2.mean_error}"""


def train_tournament(
        toolbox: base.Toolbox, ea_config: EaConfig, pset: gp.PrimitiveSet, training_dataset: pd.DataFrame,
        testing_dataset: pd.DataFrame
) -> TrainingResults:
    print("Training Model using Tournament Selection")

    # register selection operator
    toolbox.register("select", tools.selTournament, tournsize=3)

    # register fitness evaluation
    toolbox.register(
        "evaluate",
        evaluate_all_cases,
        toolbox=toolbox,
        dataset=training_dataset,
        target_var=ea_config.target_var,
        err_metric=ea_config.err_metric,
    )

    population = toolbox.population(n=ea_config.population_size)
    hof = tools.HallOfFame(1)

    pop, log = eaSimple_customized(
        population=population,
        toolbox=toolbox,
        cxpb=ea_config.crossover_rate,
        mutpb=ea_config.mutation_rate,
        ngen=ea_config.num_generations,
        stats=ea_config.multi_statistics,
        halloffame=hof,
        verbose=True,
        pset=pset,
        testing_data=testing_dataset
    )

    return TrainingResults(population=pop, logbook=log, hall_of_fame=hof)


def train_epsilon_lexicase(
        toolbox: base.Toolbox, ea_config: EaConfig, pset: gp.PrimitiveSet, training_dataset: pd.DataFrame,
        testing_dataset: pd.DataFrame
) -> TrainingResults:
    print("Training Model using Epsilon-Lexicase Selection")

    # register selection operator
    toolbox.register("select", tools.selAutomaticEpsilonLexicase)

    # register fitness evaluation
    toolbox.register(
        "evaluate",
        evaluate_all_cases,
        toolbox=toolbox,
        dataset=training_dataset,
        target_var=ea_config.target_var,
        err_metric=ea_config.err_metric,
    )

    population = toolbox.population(n=ea_config.population_size)
    hof = tools.HallOfFame(1)

    pop, log = eaSimple_customized(
        population=population,
        toolbox=toolbox,
        cxpb=ea_config.crossover_rate,
        mutpb=ea_config.mutation_rate,
        ngen=ea_config.num_generations,
        stats=ea_config.multi_statistics,
        halloffame=hof,
        testing_data=testing_dataset,
        pset=pset,
        verbose=True,
    )

    return TrainingResults(
        population=pop,
        logbook=log,
        hall_of_fame=hof,
    )


def test_model_old(
        ea_results: TrainingResults,
        ea_config: EaConfig,
        testing_data: pd.DataFrame,
        primitive_set: gp.PrimitiveSet,
) -> TestingResults:
    winner_func = gp.compile(ea_results.hall_of_fame[0], primitive_set)

    errors = []
    n = len(testing_data)

    for _, case in testing_data.iterrows():

        if ea_config.err_metric == "absolute":
            errors.append(abs(winner_func(*case[0:8:].values) - case[8:9:].values[0]))
        elif ea_config.err_metric == "squared":
            errors.append((winner_func(*case[0:8:].values) - case[8:9:].values[0]) ** 2)
        else:
            raise ValueError

    mean_err = math.fsum(errors) / n
    std_dev = pstdev(errors)

    print(
        f"Results for Testing Dataset:\n\tMean {ea_config.err_metric} error = {mean_err}\n\tStd.Dev = {std_dev}"
    )

    return TestingResults(mean_error=mean_err, std_dev=std_dev)


def separate_plots_evolution(
        attribute: str,
        statistic_function: str,
        ea_results_1: TrainingResults,
        ea_results_2: TrainingResults,
        algorithm_1: str,
        algorithm_2: str,
        directory: str,
):
    """
    create two separate plots for algorithm_1 and algorithm_2

    usage:
        attribute: "fitness" | "size"
        statistic_function: "min" | "max" | "avg" | "std"(-deviation)

    """
    gen = ea_results_1.logbook.select("gen")

    # get values
    min_1 = ea_results_1.logbook.chapters[attribute].select("min")
    max_1 = ea_results_1.logbook.chapters[attribute].select("max")
    avg_1 = ea_results_1.logbook.chapters[attribute].select("avg")
    std_1 = ea_results_1.logbook.chapters[attribute].select("std")

    min_2 = ea_results_2.logbook.chapters[attribute].select("min")
    max_2 = ea_results_2.logbook.chapters[attribute].select("max")
    avg_2 = ea_results_2.logbook.chapters[attribute].select("avg")
    std_2 = ea_results_2.logbook.chapters[attribute].select("std")

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=True)

    # get scale limits
    scale_x_min = int(min(gen))
    scale_x_max = int(max(gen))
    scale_y_min = int(min(min(min_1), min(min_2)))
    scale_y_max = int((max(min(max_1), min(max_2))) * 1.05)

    match statistic_function:
        case "min":
            var_1 = min_1
            var_2 = min_2
            description = "minimum"
        case "max":
            var_1 = max_1
            var_2 = max_2
            description = "maximum"
        case "avg":
            var_1 = avg_1
            var_2 = avg_2
            description = "average"
        case "std":
            var_1 = std_1
            var_2 = std_2
            description = "standard deviation"
        case _:
            print(f"Error: Invalid argument {statistic_function}!", file=stderr)
            return

    ax1.plot(gen, var_1, "b-", label=f"{statistic_function} {attribute}")
    ax1.set_xlabel("generation")
    ax1.set_ylabel(attribute, color="b")
    ax1.set_title(f"{algorithm_1} parent selection")
    ax1.set_xlim(scale_x_min, scale_x_max)

    ax1.legend()

    ax2.plot(gen, var_2, "b-", label=f"{statistic_function} {attribute}")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel(attribute, color="b")
    ax2.set_title(f"{algorithm_2} parent selection")
    ax2.set_xlim(scale_x_min, scale_x_max)
    ax2.set_ylim(scale_y_min, scale_y_max)

    if not attribute == "fitness" and statistic_function == "min":
        ax1.set_ylim(scale_y_min, scale_y_max)
        ax2.set_ylim(scale_y_min, scale_y_max)
    else:
        # TODO: find a better way to do this
        ax1.set_ylim(0, 1000)
        ax2.set_ylim(0, 1000)

    ax2.legend()

    fig.suptitle(f"{description} {attribute} during Training Phase")

    fpath = f"../results_50gens_500pop/{directory}/plots/{statistic_function}_{attribute}_separate.png"

    plt.savefig(fpath, dpi=100)


def combined_plots_evolution(
        attribute: str,
        statistic_function: str,
        ea_results_1: TrainingResults,
        ea_results_2: TrainingResults,
        algorithm_1: str,
        algorithm_2: str,
        directory: str,
):
    """
    create a combined plot for algorithm_1 and algorithm_2

    usage:
        attribute: "fitness" | "size"
        statistic_function: "min" | "max" | "avg" | "std"(-deviation)

    """
    gen = ea_results_1.logbook.select("gen")

    # get values
    min_1 = ea_results_1.logbook.chapters[attribute].select("min")
    max_1 = ea_results_1.logbook.chapters[attribute].select("max")
    avg_1 = ea_results_1.logbook.chapters[attribute].select("avg")
    std_1 = ea_results_1.logbook.chapters[attribute].select("std")

    min_2 = ea_results_2.logbook.chapters[attribute].select("min")
    max_2 = ea_results_2.logbook.chapters[attribute].select("max")
    avg_2 = ea_results_2.logbook.chapters[attribute].select("avg")
    std_2 = ea_results_2.logbook.chapters[attribute].select("std")

    fig, ax = plt.subplots(constrained_layout=True)

    # get scale limits
    scale_x_min = int(min(gen))
    scale_x_max = int(max(gen))
    scale_y_min = int(min(min(min_1), min(min_2)))
    scale_y_max = int((max(min(max_1), min(max_2))) * 1.1)

    match statistic_function:
        case "min":
            var_1 = min_1
            var_2 = min_2
            description = "minimum"
        case "max":
            var_1 = max_1
            var_2 = max_2
            description = "maximum"
        case "avg":
            var_1 = avg_1
            var_2 = avg_2
            description = "average"
        case "std":
            var_1 = std_1
            var_2 = std_2
            description = "standard deviation"
        case _:
            print(f"Error: Invalid argument {statistic_function}!", file=stderr)
            return

    ax.plot(gen, var_1, "b-", label=f"{algorithm_1}")
    ax.plot(gen, var_2, "r-", label=f"{algorithm_2}")
    ax.set_xlabel("Generation")
    ax.set_ylabel(attribute, color="b")
    ax.set_title(f"{description} {attribute} during Training Phase")
    ax.set_xlim(scale_x_min, scale_x_max)

    if not attribute == "fitness" and statistic_function == "min":
        ax.set_ylim(scale_y_min, scale_y_max)

    ax.legend()

    fpath = f"../results_50gens_500pop/{directory}/plots/{statistic_function}_{attribute}_combined.png"

    plt.savefig(fpath, dpi=100)


def log_single_run(
        ea_results: TrainingResults,
        testing_results: TestingResults,
        selection_method: str
):
    """
    Log parameters data from Training and Testing Phase for a single gp run
    """

    if selection_method.lower() == "epsilon-lexicase":
        directory = f'../results_50gens_500pop/epsilon_lexicase'
    elif selection_method.lower() == "tournament":
        directory = f'../results_50gens_500pop/tournament'

    try:
        os.makedirs(directory)
    except FileExistsError:
        pass

    # logging the records from the training phase
    path = f"{directory}/log.csv"
    # create csv header if file doesn't already exist
    if not os.path.exists(path):
        with open(path, "w") as ofs:
            ofs.write("gen,fit_min_training,fit_std_training,fit_avg_training,fit_testing,fit_testing_stddev\n")

    # log training results_50gens_500pop
    with open(path, "a") as ofs:
        gen = ea_results.logbook.select("gen")
        fit_min = ea_results.logbook.chapters["fitness"].select("min")
        fit_std = ea_results.logbook.chapters["fitness"].select("std")
        fit_avg = ea_results.logbook.chapters["fitness"].select("avg")
        # fit_testing = ea_results.logbook.select["testing_err"]
        # fit_testing_stddev = ea_results.logbook.select("testing_err_stddev")

        for a, b, c, d in zip(gen, fit_min, fit_std, fit_avg):
            ofs.write(f"{a},{b},{c},{d}\n")

    # logging the results_50gens_500pop from testing phase
    path = f"{directory}/testing/testing_log.csv"
    # create csv header if file doesn't already exist
    if not os.path.exists(path):
        with open(path, "w") as ofs:
            ofs.write("mse,stddev\n")
    # log testing results_50gens_500pop
    with open(path, "a") as ofs:
        ofs.write(f"{testing_results.mean_error},{testing_results.std_dev}\n")


def log_full_results(results: FullResults, filepath: str):
    """writes results_50gens_500pop (in append-mode) for best training and testing fitness for both algorithms in a full run"""
    # create header row if file doesn't exist
    if not os.path.exists(filepath):
        with open(filepath, "w") as fstr:
            fstr.write(
                f"{results.algorithm_1}_training_min_fitness,{results.algorithm_1}_testing_min_fitness,{results.algorithm_2}_training_min_fitness,{results.algorithm_2}_testing_min_fitness\n"
            )
    with open(filepath, "a") as fstr:
        fstr.write(f"{results.get_csv_str()}\n")


def eaSimple_customized(population, toolbox, cxpb, mutpb, ngen, pset: gp.PrimitiveSet, testing_data: pd.DataFrame,
                        stats=None,
                        halloffame=None, verbose=__debug__):
    """
    customized version of deap.algorithms.eaSimple() that additionally computes and logs
    the testing error for each generation's elite model
    """

    def get_testing_errors(model) -> List[float]:
        """Compute and return squared testing error"""
        winner_func = gp.compile(model, pset)

        errors = []
        n = len(testing_data)

        # for each fitness case in the testing dataset...
        for _, case in testing_data.iterrows():
            # compute squared error
            errors.append((winner_func(*case[0:8:].values) - case[8:9:].values[0]) ** 2)

        return errors

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else []) # + ["testing_err", "testing_err_stddev"]

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # compute testing error (=mse) and its std_dev for elite model
        testing_errors = get_testing_errors(model=halloffame[0])
        

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

    return population, logbook
