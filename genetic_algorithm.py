import random
import copy
from typing import List, Tuple
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Cache global
fitness_cache = {}

# Para detectar mudança de dataset
last_data_shape = None  

def gerar_modelos_randomforest(best_model):
    """
    Gera novos modelos random forest para a população inicial

    Parâmetros:
    - (best_model): Melhor modelo encontrado nos testes

    Retorno:
    Lista com todos os cinco modelos.
    """
    modelos = [best_model]

    for _ in range(4):
        # Número de árvores
        n_estimators = random.randint(50, 200)
        # Profundidade
        max_depth = random.randint(5, 21)
        # Número de features
        max_features = random.choice(['sqrt', 'log2'])
        # Fração de amostras
        max_samples = random.choice([0.5, 0.7, 0.9])

        modelo = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            max_samples=max_samples,
            n_jobs=1,
            random_state=random.randint(0, 10000)
        )

        modelos.append(modelo)

    return modelos

def calculate_fitness(model, x_train, y_train, x_test, y_test, x_initial, y_initial, cv, scoring) -> float:
    """
    Calcula o fitness de um modelo de Machine Learning baseado no cross_val_score do modelo.

    Parâmetros:
    - (model, x_train, y_train, x_test, y_test, x_initial, y_initial, cv, scoring): Todos os parâmetros do modelo usado.

    Retorno:
    float: A média acurácia.
    """
    global fitness_cache, last_data_shape

    # Verifica se o dataset mudou — se sim, limpa o cache
    current_shape = (x_initial.shape, y_initial.shape)
    if last_data_shape is None or current_shape != last_data_shape:
        fitness_cache.clear()
        last_data_shape = current_shape

    # Cria uma chave única com os hiperparâmetros do modelo
    key = tuple(sorted(model.get_params().items()))

    # Se o fitness já foi calculado para esse modelo, apenas refaz o fit e retorna o valor do cache
    if key in fitness_cache:
        model.fit(x_train, y_train)
        return fitness_cache[key]

    # Caso contrário, faz o fit e calcula o fitness normalmente
    model.fit(x_train, y_train)

    scores = cross_val_score(
        model,
        X=x_initial,
        y=y_initial,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    mean_score = scores.mean()

    # Guarda o resultado no cache
    fitness_cache[key] = mean_score
    
    # NOTE: Retorno do cálculo de fitness pelo valor de acurácia
    # predict = model.predict(x_test)
    # return accuracy_score(y_test, predict)

    # Retorna a média fitness
    return mean_score

def order_crossover(parent1, parent2) -> List:
    """
    Executa o algoritmo de cruzamento por ordenação (OX) entre dois pais sequenciais para criar a sequência de um filho.

    Parâmetros:
    - parent1: A sequência do primeiro pai.
    - parent2: A sequência do segundo pai.

    Retorno:
    A sequência do filho resultante do cruzamento por ordenação.
    """
    length = len(parent1)

    # Escolha entre dois índices aleatórios para o cruzamento
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)

    # Inicialização do filho com a cópia da substring do primeiro pai
    child = parent1[start_index:end_index]

    # Preenchimento das posições restantes com os genes do segundo pai
    remaining_positions = [
        i for i in range(length) if i < start_index or i >= end_index
    ]

    # Busca dos genes restantes para o filho
    child_param_names = {name for name, _ in child}
    remaining_genes = [gene for gene in parent2 if gene[0] not in child_param_names]

    # Looping para inserção dos genes no novo indivíduo
    for position, gene in zip(remaining_positions, remaining_genes):
        child.insert(position, gene)

    # Dicionário e modelo do filho
    child_dict = dict(child)
    child_model = RandomForestClassifier(**child_dict)

    return child_model

def mutate(
    solution: List[Tuple[float, float]], mutation_probability: float
) -> List[Tuple[float, float]]:
    """
    Mutação da solução invertendo o segmento da sequência de valores de acordo com a probabilidade de mutação.

    Parâmetros:
    - solution: A sequência de valores da solução a sofrer mutação.
    - mutation_probability: A probabilidade de mutação para cada indivíduo.

    Retorno:
    A sequência de valores com a mutação aplicada.
    """
    mutated_solution = copy.deepcopy(solution)

    # Checar se a mutação deve ocorrer
    if random.random() < mutation_probability:

        # Garantir que tenha pelo menos dois indivíduos para a troca.
        if len(solution) < 2:
            return solution

        # Seleciona um índice aleatório (exceto pelo último índice) para a troca
        index = random.randint(0, len(solution) - 2)

        # Troca do índice selecionado pelo próximo índice
        mutated_solution[index], mutated_solution[index + 1] = (
            solution[index + 1],
            solution[index],
        )

    # Dicionário e modelo do filho
    mutated_child_dict = dict(mutated_solution)
    mutated_child_model = RandomForestClassifier(**mutated_child_dict)

    return mutated_child_model

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
