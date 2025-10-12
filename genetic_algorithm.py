import random
import copy
from typing import List, Tuple
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def gerar_modelos_randomforest(best_model):
    """
    Gera novos modelos random forest para a população inicial

    Parâmetros:
    - (best_model): Melhor modelo encontrado nos testes

    Returns:
    Lista com todos os cinco modelos.
    """
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
            random_state=random.randint(0, 10000)
        )

        modelos.append(modelo)

    return modelos

def calculate_fitness(model, x_train, y_train, x_test, y_test, x_initial, y_initial, cv, scoring) -> float:
    """
    Calcula o fitness de um modelo de Machine Learning baseado no cross_val_score do modelo.

    Parâmetros:
    - (model, x_train, y_train, x_test, y_test, x_initial, y_initial, cv, scoring): Todos os parâmetros do modelo usado.

    Returns:
    float: A média acurácia.
    """
    model.fit(x_train, y_train)
    scores = cross_val_score(model, X=x_initial, y=y_initial, cv=cv, scoring=scoring)
    return scores.mean()

# NOTE: entender uso do order_crossover em nossa lógica
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

# NOTE: entender necessidade de uso do mutate em nossa lógica
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

def sort_population(
    population, fitness
) -> List:
    """
    Ordena a população baseado nos valores de fitness

    Parâmetros:
    - population: A população de soluções, onde cada solução é representada como uma lista.
    - fitness: Os valores de fitness correspondentes para cada solução na população.

    Retorno:
    A população ordenada com os valores de fitness correspondentes
    """
    # Combinação das listas em pares
    combined_lists = list(zip(population, fitness))

    # Ordenação baseada na lista de valores fitness
    sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1], reverse=True)

    sorted_population, sorted_fitness = zip(*sorted_combined_lists)
    return sorted_population, sorted_fitness
