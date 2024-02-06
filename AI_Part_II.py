import random
import matplotlib.pyplot as plt

# Genetic Algorithm Parameters
POPULATION_SIZE = 50
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.02
NUMBER_OF_GENERATIONS = 200

# Function to read problem data from a file and parse it
def read_problem_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return parse_data_to_problems(data)

# Function to convert the read data into a structured list of problem instances
def parse_data_to_problems(data):
    problems = []
    lines = data.strip().split('\n')
    index = 0
    while index < len(lines):
        # Extracting the name, item count, and capacity for each problem
        problem_name = lines[index].strip().strip("'")
        index += 1
        item_count = int(lines[index].strip())
        index += 1
        max_capacity = int(lines[index].strip())
        index += 1
        problem_items = []
        # Extracting the weight and quantity of each item
        for _ in range(item_count):
            item_weight, quantity = map(int, lines[index].strip().split())
            problem_items.append((item_weight, quantity))
            index += 1
        problems.append({'name': problem_name, 'capacity': max_capacity, 'items': problem_items})
    return problems

# Function to create an initial random population for a given problem instance
def create_initial_population(instance):
    population = []
    for _ in range(POPULATION_SIZE):
        # Generating a random solution where each item is placed in a random bin
        individual = [random.randint(1, POPULATION_SIZE) for weight, count in instance['items'] for _ in range(count)]
        random.shuffle(individual)
        population.append(individual)
    return population

# Function to evaluate the fitness of a solution based on the number of bins used and if any bin exceeds capacity
def evaluate_fitness(solution, weights, max_capacity):
    bin_weights = {}
    for idx, bin_id in enumerate(solution):
        bin_weights.setdefault(bin_id, 0)
        bin_weights[bin_id] += weights[idx]
    over_capacity_count = sum(weight > max_capacity for weight in bin_weights.values())
    return len(bin_weights) + over_capacity_count

# Function to generate offspring (next generation) from the current population
def generate_offspring(population, weights, max_capacity):
    offspring = []
    for _ in range(POPULATION_SIZE // 2):
        # Tournament selection of parents based on fitness
        parents = [min(random.sample(population, 3),
                       key=lambda individual: evaluate_fitness(individual, weights, max_capacity)) for _ in range(2)]

        child1, child2 = (parents[0][:], parents[1][:])
        # Crossover operation with a certain probability
        if random.random() < CROSSOVER_PROBABILITY:
            crossover_point = random.randint(1, len(child1) - 2)
            child1, child2 = child1[:crossover_point] + child2[crossover_point:], child2[:crossover_point] + child1[crossover_point:]

        # Mutation operation with a certain probability
        for individual in (child1, child2):
            for i in range(len(individual)):
                if random.random() < MUTATION_PROBABILITY:
                    individual[i] = random.randint(1, POPULATION_SIZE)

        offspring.extend([child1, child2])
    return offspring

# Core Genetic Algorithm function to evolve solutions for a given problem instance
def genetic_algorithm(instance):
    max_capacity = instance['capacity']
    # Flatten the list of items based on their quantity
    item_weights = [weight for weight, count in instance['items'] for _ in range(count)]
    population = create_initial_population(instance)
    fitness_progress = []

    # Evolving the population over a number of generations
    for _ in range(NUMBER_OF_GENERATIONS):
        new_population = generate_offspring(population, item_weights, max_capacity)
        # Keeping only the best individuals based on their fitness
        population = sorted(new_population, key=lambda individual: evaluate_fitness(individual, item_weights, max_capacity))[:POPULATION_SIZE]
        # Calculate and record the average fitness of the population
        avg_fitness = sum(evaluate_fitness(individual, item_weights, max_capacity) for individual in population) / POPULATION_SIZE
        fitness_progress.append(avg_fitness)

    # Returning the best solution and its fitness trend over generations
    return population[0], fitness_progress

# Function to plot the fitness trends of multiple problem instances
def plot_fitness_trends(problem_sets):
    plt.figure(figsize=(10, 6))

    for problem in problem_sets:
        # Running the GA for each problem and plotting the fitness
        _, fitness_trend = genetic_algorithm(problem)
        plt.plot(fitness_trend, label=f"{problem['name']}")

    # Plots the following, sets up title, labels and displays the legend
    plt.title("Average Best Fitness across Generations for Each Problem", color='red')
    plt.xlabel("Generation", color='red')
    plt.ylabel("Average Best Fitness", color='red')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    file_path = 'Binpacking-2.txt' # Defines Problem Data
    problems = read_problem_data(file_path) # Reads Problem data
    plot_fitness_trends(problems) # Plots the fitness trends