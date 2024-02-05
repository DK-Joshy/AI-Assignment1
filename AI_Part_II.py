import random
import matplotlib.pyplot as plt

# Genetic Algorithm Parameters
POPULATION_SIZE = 50
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.02
NUMBER_OF_GENERATIONS = 200

def read_problem_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return parse_data_to_problems(data)

def parse_data_to_problems(data):
    problems = []
    lines = data.strip().split('\n')
    index = 0
    while index < len(lines):
        problem_name = lines[index].strip().strip("'")
        index += 1
        item_count = int(lines[index].strip())
        index += 1
        max_capacity = int(lines[index].strip())
        index += 1
        problem_items = []
        for _ in range(item_count):
            item_weight, quantity = map(int, lines[index].strip().split())
            problem_items.append((item_weight, quantity))
            index += 1
        problems.append({'name': problem_name, 'capacity': max_capacity, 'items': problem_items})
    return problems

def create_initial_population(instance):
    population = []
    for _ in range(POPULATION_SIZE):
        individual = [random.randint(1, POPULATION_SIZE) for weight, count in instance['items'] for _ in range(count)]
        random.shuffle(individual)
        population.append(individual)
    return population

def evaluate_fitness(solution, weights, max_capacity):
    bin_weights = {}
    for idx, bin_id in enumerate(solution):
        bin_weights.setdefault(bin_id, 0)
        bin_weights[bin_id] += weights[idx]
    over_capacity_count = sum(weight > max_capacity for weight in bin_weights.values())
    return len(bin_weights) + over_capacity_count

# The previously defined generate_offspring and genetic_algorithm functions remain unchanged
def generate_offspring(population, weights, max_capacity):
    offspring = []
    for _ in range(POPULATION_SIZE // 2):
        parents = [min(random.sample(population, 3),
                       key=lambda individual: evaluate_fitness(individual, weights, max_capacity)) for _ in range(2)]

        child1, child2 = (parents[0][:], parents[1][:])
        if random.random() < CROSSOVER_PROBABILITY:
            crossover_point = random.randint(1, len(child1) - 2)
            child1, child2 = child1[:crossover_point] + child2[crossover_point:], child2[:crossover_point] + child1[crossover_point:]

        for individual in (child1, child2):
            for i in range(len(individual)):
                if random.random() < MUTATION_PROBABILITY:
                    individual[i] = random.randint(1, POPULATION_SIZE)

        offspring.extend([child1, child2])
    return offspring


def genetic_algorithm(instance):
    max_capacity = instance['capacity']
    item_weights = [weight for weight, count in instance['items'] for _ in range(count)]
    population = create_initial_population(instance)
    fitness_progress = []

    for _ in range(NUMBER_OF_GENERATIONS):
        new_population = generate_offspring(population, item_weights, max_capacity)
        population = sorted(new_population, key=lambda individual: evaluate_fitness(individual, item_weights, max_capacity))[:POPULATION_SIZE]
        avg_fitness = sum(evaluate_fitness(individual, item_weights, max_capacity) for individual in population) / POPULATION_SIZE
        fitness_progress.append(avg_fitness)

    return population[0], fitness_progress

def plot_fitness_trends(problem_sets):
    plt.figure(figsize=(10, 6))

    for problem in problem_sets:
        _, fitness_trend = genetic_algorithm(problem)
        plt.plot(fitness_trend, label=f"{problem['name']}")

    plt.title("Average Best Fitness across Generations for Each Problem")
    plt.xlabel("Generation")
    plt.ylabel("Average Best Fitness")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    file_path = 'Binpacking-2.txt'
    problems = read_problem_data(file_path)
    plot_fitness_trends(problems)