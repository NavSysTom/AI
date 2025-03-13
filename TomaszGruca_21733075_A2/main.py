# Student Name: Tomasz Gruca
# Student ID: 21733075

import random as rand
import matplotlib.pyplot as plt

# Always Cooperate strategy
always_cooperate = {
    'start': 'C',
    'C': 'C',
    'D': 'C',
}

# Always Defect strategy
always_defect = {
    'start': 'D',
    'C': 'D',
    'D': 'D',
}

# Tit for Tat
tit_for_tat = {
    'start': 'C',
    'C': 'C',
    'D': 'D'
}

# Random Selection strategy
def random_selection():
    return {
        'start': rand.choice(['C', 'D']),
        'C': rand.choice(['C', 'D']),
        'D': rand.choice(['C', 'D']),
    }

def initialise_population(pop_size):
    population = []
    for _ in range(pop_size):
        population.append(rand.choice([always_cooperate, always_defect, tit_for_tat, random_selection()])) # for each individual in the population size, give different strategy to
    return population

def fitness(player_strategy, opponent_strategy, rounds):
    total_score = 0
    player_move = player_strategy['start'] # get the first player move from the dictionary of the strategy you use
    opponent_move = opponent_strategy['start'] # get the first opponent move from the dictionary of the strategy the opponent use

    for _ in range(rounds):
        next_player_move = player_strategy[opponent_move]  # play the next move based of your opponents move
        next_opponent_move = opponent_strategy[player_move] # play the next move based of your move

        # evalutate scores
        if player_move == 'C' and opponent_move == 'C':
            total_score += 3
        elif player_move == 'C' and opponent_move == 'D':
            total_score += 0
        elif player_move == 'D' and opponent_move == 'C':
            total_score += 5
        elif player_move == 'D' and opponent_move == 'D':
            total_score += 1

        player_move = next_player_move # update the player moves 
        opponent_move = next_opponent_move # update the opponent moves

    return total_score / rounds # return the average score

# Tournament selection function
def tournament_selection(population, fitnesses, k):
    if k > len(population):
        k = len(population)
    return max(rand.sample(list(zip(population, fitnesses)), k), key=lambda x: x[1])[0]

# Generate new population with elitism and tournament selection
def generate_new_population(population, fitnesses, pop_size, tournament_size, elite_size):
    elite_count = int(elite_size * pop_size)
    sorted_population = [ind for ind, _ in sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)]
    new_population = sorted_population[:elite_count]  # Preserve elites

    while len(new_population) < pop_size:
        parent1 = tournament_selection(population, fitnesses, tournament_size)
        parent2 = tournament_selection(population, fitnesses, tournament_size)
        new_population.append(parent1)  # Add parent1
        new_population.append(parent2)  # Add parent2

    return new_population[:pop_size]

def genetic_algorithm(pop_size, tournament_size, elite_size, generations, opponent_strategy, opponent_name):
    population = initialise_population(pop_size)
    fitnesses = [0] * pop_size
    best_fitnesses = []
    highest_fitness = 0

    for _ in range(generations):
        for i in range(pop_size):
            fitnesses[i] = fitness(population[i], opponent_strategy, 50) # calculate the fitness of each individual in the population

        avg_fitness = sum(fitnesses) / pop_size
        best_fitnesses.append(avg_fitness)

        # Update highest fitness if the current average fitness is higher
        if avg_fitness > highest_fitness:
            highest_fitness = avg_fitness

        population = generate_new_population(population, fitnesses, pop_size, tournament_size, elite_size)

    plt.plot(best_fitnesses, label='Average Fitness')
    plt.xlabel('Generations')
    plt.ylabel('Average Fitness')
    plt.title('Fitness over Generations vs ' + opponent_name)
    plt.legend()
    plt.show()

    return highest_fitness

# Run genetic algorithm with different opponents
#genetic_algorithm(100, 5, 0.1, 100, always_cooperate, 'Always Cooperate')
#genetic_algorithm(100, 5, 0.1, 100, always_defect, 'Always Defect')
#genetic_algorithm(100, 5, 0.1, 100, tit_for_tat, 'Tit for Tat')
#genetic_algorithm(100, 5, 0.1, 100, random_selection(), 'Random Selection')


def fitness_noise(player_strategy, opponent_strategy, rounds, noise_probability):
    total_score = 0
    player_move = player_strategy['start']
    opponent_move = opponent_strategy['start']

    for _ in range(rounds):
        next_player_move = player_strategy[opponent_move]
        next_opponent_move = opponent_strategy[player_move]
             
        # Adding noise for opponents move changing it
        # Enabling this will add noise for the opponent
        #if rand.random() < noise_probability:
        #    next_opponent_move = 'D' if next_opponent_move == 'C' else 'C'

        # Adding noise for players move changing it
        # Enabling this will add noise for the player
        if rand.random() < noise_probability:
            next_player_move = 'D' if next_player_move == 'C' else 'C'

        if player_move == 'C' and opponent_move == 'C':
            total_score += 3
        elif player_move == 'C' and opponent_move == 'D':
            total_score += 0
        elif player_move == 'D' and opponent_move == 'C':
            total_score += 5
        elif player_move == 'D' and opponent_move == 'D':
            total_score += 1

        player_move = next_player_move
        opponent_move = next_opponent_move

    return total_score / rounds

def genetic_algorithm_noise(pop_size, tournament_size, elite_size, generations, opponent_strategy, opponent_name, noise_probability):
    population = initialise_population(pop_size)
    fitnesses = [0] * pop_size
    best_fitnesses = []
    highest_fitness = 0

    for _ in range(generations):
        for i in range(pop_size):
            fitnesses[i] = fitness_noise(population[i], opponent_strategy, 50, noise_probability)  # calculate the fitness of each individual in the population but with noise

        avg_fitness = sum(fitnesses) / pop_size
        best_fitnesses.append(avg_fitness)

        # Update highest fitness if the current average fitness is higher
        if avg_fitness > highest_fitness:
            highest_fitness = avg_fitness

        population = generate_new_population(population, fitnesses, pop_size, tournament_size, elite_size)

    plt.plot(best_fitnesses, label=f'Noise {noise_probability}')
    plt.xlabel('Generations')
    plt.ylabel('Average Fitness')
    plt.title('Fitness over Generations vs ' + opponent_name)
    plt.legend()
    plt.show()

    return highest_fitness

# Run genetic algorithm with different opponents with noise probability
noise_probability = [0.4, 0.6, 0.7]
for noise in noise_probability:
    genetic_algorithm_noise(100, 5, 0.1, 100, always_cooperate, 'Always Cooperate', noise)
    genetic_algorithm_noise(100, 5, 0.1, 100, always_defect, 'Always Defect', noise)
    genetic_algorithm_noise(100, 5, 0.1, 100, tit_for_tat, 'Tit for Tat', noise)
    genetic_algorithm_noise(100, 5, 0.1, 100, random_selection(), 'Random Selection', noise) 
