import pygame
import itertools
import numpy as np
import random
from genetic_algorithm import (
    calculate_fitness_ml,
    sort_population_ml,
    gerar_modelos_randomforest
)
from pygame.locals import *


# 1: Inicializando as variaveis de controle
generation_counter = itertools.count(start=1)  # Start the counter at 1
best_fitness_values = []
best_solutions = []
POPULATION_SIZE = 10
MUTATION_PROBABILITY = 0.5


# 2: Criando Looping principal
def run_genetic_algorithm(
    model, x_train, y_train, x_test, y_test, x_initial, y_initial, cv, scoring
):
    
    #População inicial
    population = gerar_modelos_randomforest(model)

    running = True
    while running:
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         running = False
        #     elif event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_q:
        #             running = False

        generation = next(generation_counter)

        # screen.fill(WHITE)        
        
        population_fitness = [calculate_fitness_ml(model, x_train, y_train, x_test, y_test, x_initial, y_initial, cv, scoring) for model in population]

        population, population_fitness = sort_population_ml(population, population_fitness)

        # best_fitness = calculate_fitness(population[0])
        # best_solution = population[0]

        # best_fitness_values.append(best_fitness)
        # best_solutions.append(best_solution)

        # # draw_plot(screen, list(range(len(best_fitness_values))),
        # #           best_fitness_values, y_label="Fitness - Distance (pxls)")

        # # draw_cities(screen, cities_locations, RED, NODE_RADIUS)
        # # draw_paths(screen, best_solution, BLUE, width=3)
        # # draw_paths(screen, population[1], rgb_color=(128, 128, 128), width=1)

        # print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)}")

        # new_population = [population[0]]  # Keep the best individual: ELITISM

        # while len(new_population) < POPULATION_SIZE:

        #     # selection
        #     # simple selection based on first 10 best solutions
        #     # parent1, parent2 = random.choices(population[:10], k=2)

        #     # solution based on fitness probability
        #     probability = 1 / np.array(population_fitness)
        #     parent1, parent2 = random.choices(population, weights=probability, k=2)

        #     # child1 = order_crossover(parent1, parent2)
        #     child1 = order_crossover(parent1, parent1)

        #     child1 = mutate(child1, MUTATION_PROBABILITY)

        #     new_population.append(child1)

        # population = new_population

        # pygame.display.flip()
        # clock.tick(FPS)
