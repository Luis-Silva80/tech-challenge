# Importação das bibliotecas necessárias
import itertools
import random
import sys
import pygame
import numpy as np
from draw_functions import draw_plot
from genetic_algorithm import (
    calculate_fitness,
    mutate,
    order_crossover,
    sort_population,
    gerar_modelos_randomforest
)

# Inicializando as variaveis de controle
generation_counter = itertools.count(start=1)
best_fitness_values = []
best_solutions = []
POPULATION_SIZE = 10
MUTATION_PROBABILITY = 0.5

# Configuração do Pygame para o gráfico
pygame.init()
PLOT_WIDTH = 800
PLOT_HEIGHT = 600
screen = pygame.display.set_mode((PLOT_WIDTH, PLOT_HEIGHT))
pygame.display.set_caption('Evolução do Algoritmo Genético')
clock = pygame.time.Clock()
FPS = 60

# Criando lopping principal do algoritmo genético
def run_genetic_algorithm(
    model, x_train, y_train, x_test, y_test, x_initial, y_initial, cv, scoring
):

    # Geração da população inicial
    population = gerar_modelos_randomforest(model)

    # Lógica para o algoritmo rodar interminavelmente
    running = True

    # Looping para o cálculo das próximas gerações
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        generation = next(generation_counter)

        # Capturando todas as médias de acurácia da população
        population_fitness = []
        for model in population:
            population_fitness.append(calculate_fitness(
                model, x_train, y_train, x_test, y_test, x_initial, y_initial, cv, scoring))

        # Sorteio da população para ordenar os melhores resultados
        population, population_fitness = sort_population(
            population, population_fitness)

        # Armazenar os melhores resultados da solução e valor de fitness
        best_fitness = population_fitness[0]
        best_solution = population[0]

        # Armazenar os melhores resultados encontrados para distribuição em gráfico
        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)

        # Atualizar o gráfico
        generations = list(range(1, len(best_fitness_values) + 1))
        draw_plot(screen, generations, best_fitness_values, 
                     x_label='Geração', y_label='Fitness', use_cache=True)
        pygame.display.flip()
        clock.tick(FPS)

        # Printar a geração e o melhor valor de fitness
        print(f"Geração {generation}: Melhor fitness = {best_fitness:.3f}")
        new_population = [population[0]]  # Manter o melhor indivíduo: ELITISMO

        # Algoritmo de seleção para cruzamento dos novos indivíduos
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

    # Encerrar algoritmo
    pygame.quit()
    sys.exit()