# AI-Assignment1
For all three problems, we represent solutions as binary strings of a fixed length (30 in this case). Each bit in the string represents a decision or a feature.
## 1.1 One-Max Problem
For the One-Max Problem, the fitness function calculates the fitness of a solution by counting the number of '1's in the binary string. The fitness is directly proportional to the number of '1's present in the string.
## 1.2 Evolving to a Target String
For the Evolving to a Target String problem, we define a target binary string. The fitness function calculates the fitness of a solution by counting the number of matching bits between the current binary string and the target string. The more matching bits, the higher the fitness.
## 1.3 Deceptive Landscape
For the Deceptive Landscape problem, the fitness function is modified. It counts the number of '1's in the string, but if there are no '1's present, the fitness is set to be 2 times the length of the solution. This modification creates a deceptive landscape, making it challenging for the genetic algorithm to find the optimal solution.
## Selection
We use tournament selection for selecting parents for crossover. In each generation, we randomly select a subset of individuals (tournament size = 5) and choose the fittest individual from that subset as a parent.
## Crossover
We perform one-point crossover, where two parents are selected, and a random crossover point is chosen. The children are created by swapping the bits of the parents before and after the crossover point.
## Mutation
Mutation is applied with a low probability (mutation rate = 0.01) for each bit in an individual's binary string. If mutation occurs for a bit, it is flipped (from '0' to '1' or from '1' to '0').

# Plots of Performance
## 1.1 One-Max Plotting
![Alt](https://github.com/EddieSheehy/AI-Assignment1/blob/main/partaPhotos/1.1_photo.png)

## 1.2 Evolving to a Target String Plotting
![Alt](https://github.com/EddieSheehy/AI-Assignment1/blob/main/partaPhotos/1.2_photo.png)

## 1.3 Deceptive Landscape Plotting
![Alt](https://github.com/EddieSheehy/AI-Assignment1/blob/main/partaPhotos/1.3_photo.png)

# Description of Results

## 1.1 One-Max Problem
The genetic algorithm quickly converges to the optimal solution, which is a binary string with all '1's. The average fitness increases steadily over generations, demonstrating the algorithm's effectiveness in a simple search landscape.

## 1.2 Evolving to a Target String
The algorithm successfully evolves the population to match the target binary string. The average fitness increases over generations as the algorithm finds better solutions, demonstrating its ability to search for specific patterns.

## 1.3 Deceptive Landscape
The Deceptive Landscape poses a significant challenge to the genetic algorithm. The fitness landscape is deceptive, with the optimal solution not containing any '1's. The algorithm struggles to find the optimal solution, and the average fitness remains high throughout the generations.

In summary, the genetic algorithm successfully addresses the three different optimization problems, showcasing its adaptability to different fitness landscapes. It performs well in simple problems but faces challenges in deceptive landscapes where the optimal solution is not straightforward.
