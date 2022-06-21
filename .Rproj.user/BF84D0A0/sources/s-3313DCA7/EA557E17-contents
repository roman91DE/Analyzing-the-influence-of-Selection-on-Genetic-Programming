import math
import operator
import os
from copy import deepcopy
from dataclasses import dataclass
from random import random, randint
from sys import argv
from typing import Any, Tuple, Callable
from statistics import mean as statistics_mean

import numpy as np
import pandas as pd
from deap import gp, tools, creator, base, algorithms
from sklearn.model_selection import train_test_split as skl_train_test_split

"""

Description:
    This program is used to run a single run of the basic experiment.
    Two models are trained and tested, one using Tournament Selection and the other using epsilon-Lexicase Selection.

Output:
    Results for both algorithms are written to ../results/single_run/<algorithm>/<num>.tsv

Arguments:
    num: Integer to enumerate which run the results are associated with
    
Conda Environment:
    Specified in the file <gp_research.yml> in the same directory

"""


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

    def __split(__training_split: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_excel(f"{DESTINATION}/{FILENAME}")
        return skl_train_test_split(
            df,
            train_size=__training_split,
            test_size=(1.0 - __training_split),
        )

    __download()
    return __split(training_split)


def get_primitive_set() -> gp.PrimitiveSet:
    """returns a basic deap.gp.PrimitiveSet object for symbolic regression on the dataset without registering methods
    for selection and fitness evaluation"""

    # terminals
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


def get_toolbox(primitive_set: gp.PrimitiveSet) -> base.Toolbox:
    """returns a basic deap.base.Toolbox object for symbolic regression"""

    # create custom types
    # min fitness object: objective: minimize MSE for y1^
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


def evaluate_single_case(func: Callable, case: pd.Series) -> float:
    """
    Evaluates an individual, compiled program for a single fitness case,
    computes and returns squared error for prediction and empirical value
    """
    # compute individual with case variables
    predicted_value_y1 = func(*case[0:8:].values)
    # get empirical value for y1
    empirical_value_y1 = case.values[8]
    # compute and return squared error
    return (predicted_value_y1 - empirical_value_y1) ** 2


def evaluate_all_cases(
        individual: "creator.Individual",
        toolbox: base.Toolbox,
        dataset: pd.core.frame.DataFrame,
) -> Any:
    """
    Evaluates an individual program for all fitness cases inside the dataset,
    computes and returns the MSE
    """
    # compile the tree expression into a callable function
    compiled_individual = toolbox.compile(expr=individual)

    squared_errors = np.zeros(shape=len(dataset), dtype=np.double)

    # iterate through all fitness cases and aggregate squared errors
    for idx, (_, fitness_case) in enumerate(dataset.iterrows()):
        squared_errors[idx] = evaluate_single_case(
            func=compiled_individual, case=fitness_case
        )

    # compute and return mean error
    return np.mean(squared_errors),


def get_statistics_obj() -> tools.MultiStatistics:
    """returns a deap.tools.Statics object"""
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_fit.register("avg", np.mean)
    stats_fit.register("std", np.std)
    stats_fit.register("min", np.min)
    stats_fit.register("max", np.max)

    return stats_fit


@dataclass
class EaConfig:
    """wrapper for basic EA configurations"""

    population_size: int
    num_generations: int
    mutation_rate: float
    crossover_rate: float
    multi_statistics: tools.Statistics
    elite_size: int
    tournament_size: int

    def log_string(self):
        return f"""population_size: {self.population_size}
num_generations: {self.num_generations}
mutation_rate: {self.mutation_rate}
crossover_rate: {self.crossover_rate}
elite_size: {self.elite_size}
tournament_size: {self.tournament_size}
EOF
"""


def run_tournament(
        toolbox: base.Toolbox,
        ea_config: EaConfig,
        pset: gp.PrimitiveSet,
        training_dataset: pd.DataFrame,
        testing_dataset: pd.DataFrame,
) -> tools.Logbook:
    print("Training Model using Tournament Selection")


    toolbox.register("select", tools.selTournament, tournsize=ea_config.tournament_size)

    # register fitness evaluation for training
    toolbox.register(
        "evaluate",
        evaluate_all_cases,
        toolbox=toolbox,
        dataset=training_dataset,
    )

    population = toolbox.population(n=ea_config.population_size)
    hof = tools.HallOfFame(1)

    log = evolutionaryAlgorithm(
        population=population,
        toolbox=toolbox,
        cxpb=ea_config.crossover_rate,
        mutpb=ea_config.mutation_rate,
        ngen=ea_config.num_generations,
        stats=ea_config.multi_statistics,
        halloffame=hof,
        verbose=True,
        pset=pset,
        testing_data=testing_dataset,
    )

    return log


def run_epsilon_lexicase(
        toolbox: base.Toolbox,
        ea_config: EaConfig,
        pset: gp.PrimitiveSet,
        training_dataset: pd.DataFrame,
        testing_dataset: pd.DataFrame,
) -> tools.Logbook:
    print("Training Model using Epsilon-Lexicase Selection")

    # register selection operator
    toolbox.register("select", tools.selAutomaticEpsilonLexicase)

    # register fitness evaluation
    toolbox.register(
        "evaluate",
        evaluate_all_cases,
        toolbox=toolbox,
        dataset=training_dataset,
    )

    population = toolbox.population(n=ea_config.population_size)
    hof = tools.HallOfFame(1)

    log = evolutionaryAlgorithm(
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

    return log


def evolutionaryAlgorithm(
        population,
        toolbox,
        cxpb,
        mutpb,
        ngen,
        pset: gp.PrimitiveSet,
        testing_data: pd.DataFrame,
        stats=None,
        halloffame=None,
        verbose=__debug__,
):
    """
    customized version of deap.algorithms.eaSimple() that additionally computes and logs
    the testing error for each generation's elite model
    """

    def get_testing_errors(model) -> np.ndarray:
        """Compute and return squared testing error"""
        winner_func = gp.compile(model, pset)

        errors = np.zeros(shape=len(testing_data), dtype=np.double)

        # for each fitness case in the testing dataset...
        for idx, (_, case) in enumerate(testing_data.iterrows()):
            # compute squared error
            errors[idx] = (winner_func(*case[0:8:].values) - case[8:9:].values[0]) ** 2

        return errors

    logbook = tools.Logbook()
    logbook.header = (
            ["gen", "nevals"]
            + (stats.fields if stats else [])
            + ["testing_error", "std_testing_error"]
            + ["avg_size", "elite_size"]
    )

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

        # Replace the current population by the offspring
        population[:] = offspring

        # compute testing error (=mse) and its std_dev for elite model
        best_model = tools.selBest(population, 1)[0]
        testing_errors = get_testing_errors(model=best_model)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}

        # manually add stats for testing error and size
        record["testing_error"] = np.mean(testing_errors)
        record["std_testing_error"] = np.std(testing_errors)
        record["avg_size"] = statistics_mean([len(ind) for ind in population])
        record["elite_size"] = len(best_model)

        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        if verbose:
            print(logbook.stream)

    return logbook


def single_run(nth_run: int):
    """Train, Test and log results for both selection operators"""

    def log2file(log: tools.Logbook, algorithm: str, num: int):
        PATH = f"../results/single_run/{algorithm}"
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        with open(f"{PATH}/{str(num)}.tsv", "w") as fstream:
            fstream.write(str(log))

    # get database and split into two pd.DataFrames
    TRAINING_DATASET, TESTING_DATASET = get_datasets(training_split=0.5)

    # primitive set
    PRIMITIVE_SET = get_primitive_set()

    # deap toolbox
    TOOLBOX = get_toolbox(
        primitive_set=PRIMITIVE_SET,
    )

    # deap multi statistics
    MULTI_STATISTICS = get_statistics_obj()

    # gp configuration
    EA_CONFIG = EaConfig(
        population_size=500,
        num_generations=100,
        mutation_rate=0.2,
        crossover_rate=0.8,
        multi_statistics=MULTI_STATISTICS,
        elite_size=0,
        tournament_size=3
    )

    # train model for y1 using tournament selection
    log_tournament = run_tournament(
        toolbox=deepcopy(TOOLBOX),
        ea_config=EA_CONFIG,
        training_dataset=TRAINING_DATASET,
        testing_dataset=TESTING_DATASET,
        pset=deepcopy(PRIMITIVE_SET),
    )
    log2file(log_tournament, "tournament", nth_run)

    # train model for y1 using epsilon lexicase selection
    log_e_lexicase = run_epsilon_lexicase(
        toolbox=deepcopy(TOOLBOX),
        ea_config=EA_CONFIG,
        training_dataset=TRAINING_DATASET,
        testing_dataset=TESTING_DATASET,
        pset=deepcopy(PRIMITIVE_SET),
    )
    log2file(log_e_lexicase, "e_lexicase", nth_run)


if __name__ == "__main__":
    single_run(int(argv[1]))
