import random
import time
import numpy as np
import matplotlib.pyplot as plt

# Function to load TSP data from a file
def load_tsp(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    
    nodes = []
    reading_nodes = False
    for line in lines:
        if "NODE_COORD_SECTION" in line:
            reading_nodes = True
            continue
        if "EOF" in line:
            break
        if reading_nodes:
            parts = line.strip().split()
            nodes.append((float(parts[1]), float(parts[2])))
    
    return np.array(nodes)

# Function to compute the distance matrix for a set of nodes
def compute_distance_matrix(nodes):
    return np.linalg.norm(nodes[:, np.newaxis] - nodes, axis=2)

# Fitness function to evaluate the total distance of a tour
def fitness(individual, dist_matrix):
    path_distances = dist_matrix[individual, np.roll(individual, -1)]
    return -np.sum(path_distances)

# Function to initialize the population with random tours
def initialize_population(size, num_cities):
    population = []
    for _ in range(size):
        individual = random.sample(range(1, num_cities), num_cities - 1)
        individual.insert(0, 0)  # Ensure the tour starts and ends at the first city
        population.append(individual)
    return population

# Tournament selection function to select parents for crossover
def tournament_selection(population, fitnesses, k):
    return max(random.sample(list(zip(population, fitnesses)), k), key=lambda x: x[1])[0]

# Order crossover function to generate offspring
def order_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child1_middle = parent1[start:end]
    child2_middle = parent2[start:end]
    
    child1 = [gene for gene in parent2 if gene not in child1_middle]
    child2 = [gene for gene in parent1 if gene not in child2_middle]
    
    child1 = child1[:start] + child1_middle + child1[start:]
    child2 = child2[:start] + child2_middle + child2[start:]
    
    return child1, child2

# PMX crossover function to generate offspring
def pmx_crossover(parent1, parent2):
    size = len(parent1)
    child1, child2 = [-1]*size, [-1]*size
    start, end = sorted(random.sample(range(size), 2))
    
    child1[start:end], child2[start:end] = parent1[start:end], parent2[start:end]
    
    def pmx_fill(child, parent):
        for i in range(start, end):
            if parent[i] not in child:
                pos = i
                while start <= pos < end:
                    pos = parent.index(child[pos])
                child[pos] = parent[i]
        for i in range(size):
            if child[i] == -1:
                child[i] = parent[i]
        return child
    
    return pmx_fill(child1, parent2), pmx_fill(child2, parent1)

# Displacement mutation function to introduce variations
def displacement_mutation(individual):
    start, end = sorted(random.sample(range(1, len(individual)), 2))
    segment = individual[start:end]
    individual = individual[:start] + individual[end:]
    insert_pos = random.randint(1, len(individual) - 1)
    individual = individual[:insert_pos] + segment + individual[insert_pos:]
    return individual

# Inversion mutation function to introduce variations
def inversion_mutation(individual):
    idx1, idx2 = sorted(random.sample(range(1, len(individual)), 2))
    individual[idx1:idx2] = reversed(individual[idx1:idx2])
    return individual

# Main genetic algorithm function
def genetic_algorithm(filename, pop_size=100, generations=400, crossover_rate=0.8, mutation_rate=0.02, elite_size=2, tournament_size=3, crossover_operator=order_crossover, mutation_operator=displacement_mutation, plot=False):
    # Load TSP data and compute distance matrix
    nodes = load_tsp(filename)
    dist_matrix = compute_distance_matrix(nodes)
    
    # Initialize population
    population = initialize_population(pop_size, len(nodes))
    best_fitness_over_time = []
    global_best_fitness = float('-inf')
    no_improvement_counter = 0
    max_no_improvement_generations = 1000
    
    # Determine crossover and mutation operator names
    crossover_name = "order crossover" if crossover_operator == order_crossover else "pmx crossover"
    if mutation_operator == displacement_mutation:
        mutation_name = "displacement mutation"
    else:
        mutation_name = "inversion mutation"
    
    print(f"Using {crossover_name} and {mutation_name} with tournament size {tournament_size}")
    
    start_time = time.time()
    for gen in range(generations):
        # Evaluate fitness of the population
        fitnesses = [fitness(ind, dist_matrix) for ind in population]
        
        # Generate new population
        population = generate_new_population(population, fitnesses, pop_size, crossover_rate, mutation_rate, elite_size, crossover_operator, mutation_operator, tournament_size)
        
        # Find the best individual in the current generation
        best_individual, best_gen_fitness = max(zip(population, fitnesses), key=lambda x: x[1])
        best_fitness_over_time.append(best_gen_fitness)
        print(f"Generation {gen}: Best Fitness = {-best_gen_fitness:.2f}")
        
        # Update global best fitness and solution
        if best_gen_fitness > global_best_fitness:
            global_best_fitness = best_gen_fitness
            global_best_solution = best_individual[:]
            no_improvement_counter = 0  
        else:
            no_improvement_counter += 1
        
        # Stop if no improvement for a certain number of generations
        if no_improvement_counter >= max_no_improvement_generations:
            print(f"No improvement in fitness for {max_no_improvement_generations} generations. Stopping at generation {gen}.")
            break
    
    print(f"Best Solution Found: {-global_best_fitness:.2f} in {time.time() - start_time:.2f} sec")
    
    # Plot fitness over generations if requested
    if plot:
        plt.plot(best_fitness_over_time)
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.title("Fitness Over Generations")
        plt.show()
    
    return global_best_solution, -global_best_fitness, pop_size, generations, crossover_rate, mutation_rate, elite_size, tournament_size, crossover_name, mutation_name

# Function to generate a new population
def generate_new_population(population, fitnesses, pop_size, crossover_rate, mutation_rate, elite_size, crossover_operator, mutation_operator, tournament_size):
    # Sort population by fitness in descending order and preserve elites
    elite_count = int(elite_size * pop_size)
    sorted_population = [ind for ind, _ in sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)]
    new_population = sorted_population[:elite_count]  # Preserve elites
    
    # Generate new individuals until the population is filled
    while len(new_population) < pop_size:
        parent1, parent2 = tournament_selection(population, fitnesses, tournament_size), tournament_selection(population, fitnesses, tournament_size)
        
        # Apply crossover
        if random.random() < crossover_rate:
            child1, child2 = crossover_operator(parent1, parent2)
        else:
            child1, child2 = parent1[:], parent2[:]
        
        # Apply mutation
        child1 = mutation_operator(child1) if random.random() < mutation_rate else child1
        child2 = mutation_operator(child2) if random.random() < mutation_rate else child2
        
        new_population.extend([child1, child2])
    
    return new_population[:pop_size]

# Function to evaluate different parameter combinations
def evaluate_parameters(filename):
    parameter_combinations = [
        
        # (pop_size, generations, mutation_rate, elite_size, tournament_size, crossover_operator, mutation_operator)
        (1500, 500000, 0.02, 0.02, 3, pmx_crossover, displacement_mutation),
        (500, 50000, 0.5, 0.02, 6, order_crossover, inversion_mutation),
    ]
    
    results = []
    
    for pop_size, generations, mutation_rate, elite_size, tournament_size, crossover_operator, mutation_operator in parameter_combinations:
        print(f"Testing parameters: pop_size={pop_size}, generations={generations}, mutation_rate={mutation_rate}, elite_size={elite_size}, tournament_size={tournament_size}")
        best_solution, best_fitness, pop_size, generations, _, mutation_rate, elite_size, tournament_size, crossover_name, mutation_name = genetic_algorithm(filename, pop_size, generations, 1.0, mutation_rate, elite_size, tournament_size, crossover_operator, mutation_operator)
        results.append((best_fitness, pop_size, generations, 1.0, mutation_rate, elite_size, tournament_size, best_solution, crossover_name, mutation_name))
    
    # Find the best parameter combination
    best_result = min(results, key=lambda x: x[0])
    best_fitness, best_pop_size, best_generations, best_crossover_rate, best_mutation_rate, best_elite_size, best_tournament_size, best_solution, best_crossover_name, best_mutation_name = best_result
    
    print(f"Best parameters: Population Size = {best_pop_size}, Generations = {best_generations}, Crossover Rate = {best_crossover_rate}, Mutation Rate = {best_mutation_rate}, Elite Size = {best_elite_size}, Tournament Size = {best_tournament_size}")
    print(f"Best fitness: {best_fitness:.2f}")
    
    # Write the best parameters and fitness to a file
    with open("best_solution_berlin.txt", "a") as f:
        f.write(f"\nOverall Best Fitness: {best_fitness:.2f}\n")
        f.write(f"Overall Best Optimal Path: {best_solution + [best_solution[0]]}\n")
        f.write(f"Overall Best Parameters: Population Size = {best_pop_size}, Generations = {best_generations}, Crossover Rate = {best_crossover_rate}, Mutation Rate = {best_mutation_rate}, Elite Size = {best_elite_size}, Tournament Size = {best_tournament_size}\n")
        f.write(f"Crossover Operator: {best_crossover_name}\n")
        f.write(f"Mutation Operator: {best_mutation_name}\n")
    
    return best_result

tsp_filename = "pr1002.tsp"
best_result = evaluate_parameters(tsp_filename)

# Use the best parameters to run the genetic algorithm and plot the results
best_fitness, best_pop_size, best_generations, best_crossover_rate, best_mutation_rate, best_elite_size, best_tournament_size, best_solution, best_crossover_name, best_mutation_name = best_result
best_solution, best_fitness, _, _, _, _, _, _, _, _ = genetic_algorithm(
    tsp_filename, 
    pop_size=best_pop_size, 
    generations=best_generations, 
    crossover_rate=best_crossover_rate, 
    mutation_rate=best_mutation_rate,
    elite_size=best_elite_size,
    tournament_size=best_tournament_size,
    crossover_operator=order_crossover,
    mutation_operator=displacement_mutation,
    plot=True
)

# Print the best parameters and the best solution
print(f"Best Parameters: Population Size = {best_pop_size}, Generations = {best_generations}, Crossover Rate = {best_crossover_rate}, Mutation Rate = {best_mutation_rate}, Elite Size = {best_elite_size}, Tournament Size = {best_tournament_size}")
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness:.2f}")