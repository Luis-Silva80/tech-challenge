import random
import math
import copy
from typing import List, Tuple
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def gerar_modelos_randomforest(best_model):
    modelos = [best_model]

    for _ in range(4):
        n_estimators = random.randint(50, 200)        # Número de árvores
        max_depth = random.choice([None] + list(range(5, 21)))  # Profundidade ou None
        max_features = random.choice(['sqrt', 'log2', None])    # Número de features
        max_samples = random.choice([None, 0.5, 0.7, 0.9])      # Fração de amostras (ou None)

        modelo = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            max_samples=max_samples,
            random_state=random.randint(0, 10000)  # Para reprodutibilidade
        )

        modelos.append(modelo)

    return modelos

def calculate_fitness(path: List[Tuple[float, float]]) -> float:
    """
    Calculate the fitness of a given path based on the total Euclidean distance.

    Parameters:
    - path (List[Tuple[float, float]]): A list of tuples representing the path,
      where each tuple contains the coordinates of a point.

    Returns:
    float: The total Euclidean distance of the path.
    """
    # distance = 0
    # n = len(path)
    # for i in range(n):
    #     distance += calculate_distance(path[i], path[(i + 1) % n])

    # return distance


def calculate_fitness_ml(model, x_train, y_train, x_test, y_test, x_initial, y_initial, cv, scoring) -> float:
    """
    Calculates the fitness of a Machine Learning model based on the model's cross_val_score.

    Parameters:
    - (model, x_train, y_train, x_test, y_test, x_initial, y_initial, cv, scoring): All parameters of the model used.

    Returns:
    float: The average accuracy
    """
    model.fit(x_train, y_train)
    predict_model = model.predict(x_test)

    scores = cross_val_score(model, x_initial, y_initial, cv, scoring)
    scores.mean()

def order_crossover(
    parent1: List[Tuple[float, float]], parent2: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Perform order crossover (OX) between two parent sequences to create a child sequence.

    Parameters:
    - parent1 (List[Tuple[float, float]]): The first parent sequence.
    - parent2 (List[Tuple[float, float]]): The second parent sequence.

    Returns:
    List[Tuple[float, float]]: The child sequence resulting from the order crossover.
    """
    length = len(parent1)

    # Choose two random indices for the crossover
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)

    # Initialize the child with a copy of the substring from parent1
    child = parent1[start_index:end_index]

    # Fill in the remaining positions with genes from parent2
    remaining_positions = [
        i for i in range(length) if i < start_index or i >= end_index
    ]
    remaining_genes = [gene for gene in parent2 if gene not in child]

    for position, gene in zip(remaining_positions, remaining_genes):
        child.insert(position, gene)

    return child


### demonstration: crossover test code
# Example usage:
# parent1 = [(1, 1), (2, 2), (3, 3), (4,4), (5,5), (6, 6)]
# parent2 = [(6, 6), (5, 5), (4, 4), (3, 3),  (2, 2), (1, 1)]

# # parent1 = [1, 2, 3, 4, 5, 6]
# # parent2 = [6, 5, 4, 3, 2, 1]


# child = order_crossover(parent1, parent2)
# print("Parent 1:", [0, 1, 2, 3, 4, 5, 6, 7, 8])
# print("Parent 1:", parent1)
# print("Parent 2:", parent2)
# print("Child   :", child)


# # Example usage:
# population = generate_random_population(5, 10)

# print(calculate_fitness(population[0]))


# population = [(random.randint(0, 100), random.randint(0, 100))
#           for _ in range(3)]


# TODO: implement a mutation_intensity and invert pieces of code instead of just swamping two.
def mutate(
    solution: List[Tuple[float, float]], mutation_probability: float
) -> List[Tuple[float, float]]:
    """
    Mutate a solution by inverting a segment of the sequence with a given mutation probability.

    Parameters:
    - solution (List[int]): The solution sequence to be mutated.
    - mutation_probability (float): The probability of mutation for each individual in the solution.

    Returns:
    List[int]: The mutated solution sequence.
    """
    mutated_solution = copy.deepcopy(solution)

    # Check if mutation should occur
    if random.random() < mutation_probability:

        # Ensure there are at least two cities to perform a swap
        if len(solution) < 2:
            return solution

        # Select a random index (excluding the last index) for swapping
        index = random.randint(0, len(solution) - 2)

        # Swap the cities at the selected index and the next index
        mutated_solution[index], mutated_solution[index + 1] = (
            solution[index + 1],
            solution[index],
        )

    return mutated_solution


### Demonstration: mutation test code
# # Example usage:
# original_solution = [(1, 1), (2, 2), (3, 3), (4, 4)]
# mutation_probability = 1

# mutated_solution = mutate(original_solution, mutation_probability)
# print("Original Solution:", original_solution)
# print("Mutated Solution:", mutated_solution)


def sort_population(
    population: List[List[Tuple[float, float]]], fitness: List[float]
) -> Tuple[List[List[Tuple[float, float]]], List[float]]:
    """
    Sort a population based on fitness values.

    Parameters:
    - population (List[List[Tuple[float, float]]]): The population of solutions, where each solution is represented as a list.
    - fitness (List[float]): The corresponding fitness values for each solution in the population.

    Returns:
    Tuple[List[List[Tuple[float, float]]], List[float]]: A tuple containing the sorted population and corresponding sorted fitness values.
    """
    # Combine lists into pairs
    combined_lists = list(zip(population, fitness))

    # Sort based on the values of the fitness list
    sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1])

    # Separate the sorted pairs back into individual lists
    sorted_population, sorted_fitness = zip(*sorted_combined_lists)

    return sorted_population, sorted_fitness

def sort_population_ml(
    population, fitness
) -> List:
    """
    Sort a population based on fitness values.

    Parameters:
    - population (List[List[Tuple[float, float]]]): The population of solutions, where each solution is represented as a list.
    - fitness (List[float]): The corresponding fitness values for each solution in the population.

    Returns:
    Tuple[List[List[Tuple[float, float]]], List[float]]: A tuple containing the sorted population and corresponding sorted fitness values.
    """
    # Combine lists into pairs
    combined_lists = list(population, fitness)

    # Sort based on the values of the fitness list
    sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1])

    return sorted_combined_lists
