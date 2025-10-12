# 1: Importação das bibliotecas necessárias
import itertools
import random
import numpy as np
from genetic_algorithm import (
    calculate_fitness,
    mutate,
    order_crossover,
    sort_population,
    gerar_modelos_randomforest
)

# 2: Inicializando as variaveis de controle
generation_counter = itertools.count(start=1)
best_fitness_values = []
best_solutions = []
POPULATION_SIZE = 10
MUTATION_PROBABILITY = 0.5

# 2: Criando lopping principal do algoritmo genético
def run_genetic_algorithm(
    model, x_train, y_train, x_test, y_test, x_initial, y_initial, cv, scoring
):

    # 3: Geração da população inicial
    population = gerar_modelos_randomforest(model)

    # 4: Lógica para o algoritmo rodar interminavelmente
    running = True

    # 5: Looping para o cálculo das próximas gerações
    while running:
        generation = next(generation_counter)

        # 6: Capturando todas as médias de acurácia da população
        population_fitness = []
        for model in population:
            population_fitness.append(calculate_fitness(
                model, x_train, y_train, x_test, y_test, x_initial, y_initial, cv, scoring))

        # 7: Sorteio da população para ordenar os melhores resultados
        population, population_fitness = sort_population(
            population, population_fitness)

        # 8 Armazenar os melhores resultados da solução e valor de fitness
        best_fitness = population_fitness[0]
        best_solution = population[0]

        # 9 Armazenar os melhores resultados encontrados para distribuição em gráfico
        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)

        # TODO: Montar os gráficos
        # draw_plot(screen, list(range(len(best_fitness_values))),
        # best_fitness_values, y_label="Fitness - Distance (pxls)")
        # draw_cities(screen, cities_locations, RED, NODE_RADIUS)
        # draw_paths(screen, best_solution, BLUE, width=3)
        # draw_paths(screen, population[1], rgb_color=(128, 128, 128), width=1)

        # : Printar a geração e o melhor valor de fitness
        print(f"Geração {generation}: Melhor fitness = {best_fitness:.4f}")
        new_population = [population[0]]  # Manter o melhor indivíduo: ELITISMO

        # : Algoritmo de seleção para cruzamento dos novos indivíduos
        while len(new_population) < POPULATION_SIZE:
            # Solução baseada na probabilidade de fitness
            probability = 1 / np.array(population_fitness)
            parent1, parent2 = random.choices(population, weights=probability, k=2)

            # Dicionário dos hiperparâmetros de cada indivíduo pai
            dict_parent1: dict = parent1.get_params()
            dict_parent2: dict = parent2.get_params()

            # Lista dos hiperparâmetros para o cruzamento
            list_parent1 = list(dict_parent1.items())
            list_parent2 = list(dict_parent2.items())

            # Cruzamento dos indivíduos pai para geração do filho
            child = order_crossover(list_parent1, list_parent2)

            # Dicionário e lista dos hiperparâmetros do filho
            dict_child: dict = child.get_params()
            list_child = list(dict_child.items())

            # Mutação do indivíduo filho
            child = mutate(list_child, MUTATION_PROBABILITY)

            # Anexo do filho a nova população
            new_population.append(child)

        # Substituição da nova população
        population = new_population

        # pygame.display.flip()
        # clock.tick(FPS)
