# 1: Importação das bibliotecas necessárias
import itertools
import random
import numpy as np
from genetic_algorithm import (
    calculate_fitness,
    mutate,
    order_crossover_ml,
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

    # 2.1: Geração da população inicial
    population = gerar_modelos_randomforest(model)

    # 2.2: Condição de término do algoritmo
    running = True
    while running:
        # NOTE: pensar qual será a condição de término do programa (100% de acurácia?)
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         running = False
        #     elif event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_q:
        #             running = False

        # NOTE: entender a necessidade da lógica abaixo
        generation = next(generation_counter)

        # 2.3: Capturando todas as médias de acurácia da população
        population_fitness = []
        for model in population:
            population_fitness.append(calculate_fitness(
                model, x_train, y_train, x_test, y_test, x_initial, y_initial, cv, scoring))

        # 2.4: Sorteio da população para ordenar os melhores resultados
        population, population_fitness = sort_population(
            population, population_fitness)

        # 2.5 Armazenar os melhores resultados da solução e valor de fitness
        best_fitness = population_fitness[0]
        best_solution = population[0]

        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)

        # draw_plot(screen, list(range(len(best_fitness_values))),
        #           best_fitness_values, y_label="Fitness - Distance (pxls)")

        # draw_cities(screen, cities_locations, RED, NODE_RADIUS)
        # draw_paths(screen, best_solution, BLUE, width=3)
        # draw_paths(screen, population[1], rgb_color=(128, 128, 128), width=1)

        print(f"Generation {generation}: Best fitness = {best_fitness:.12f}")
        new_population = [population[0]]  # Manter o melhor indivíduo: ELITISMO

        while len(new_population) < POPULATION_SIZE:
            # selection
            # simple selection based on first 10 best solutions
            # parent1, parent2 = random.choices(population[:10], k=2)

            # solution based on fitness probability
            probability = 1 / np.array(population_fitness)
            parent1, parent2 = random.choices(population, weights=probability, k=2)

            dict_parent1: dict = parent1.get_params()
            dict_parent2: dict = parent2.get_params()

            list_parent1 = list(dict_parent1.items())
            list_parent2 = list(dict_parent2.items())

            child1 = order_crossover_ml(list_parent1, list_parent2)
            dict_child: dict = child1.get_params()
            list_child = list(dict_child.items())

            child1 = mutate(list_child, MUTATION_PROBABILITY)

            new_population.append(child1)

        population = new_population


        # pygame.display.flip()
        # clock.tick(FPS)
