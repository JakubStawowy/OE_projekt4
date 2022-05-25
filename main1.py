import random
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from crossover import arithmetic_crossover, linear_crossover, blend_crossover_alpha, blend_crossover_alpha_beta, \
    average_crossover
from svc import SVCParameters, SVCParametersFitness, mutationSVC

realRepresentation = True
minimum = True

pd.set_option('display.max_columns', None)
df = pd.read_csv("data.csv", sep=',')
# df = pd.read_csv("data.csv", sep=r'\s*,\s*')
# print(df.columns.tolist())
y = df['Status']
df.drop('Status', axis=1, inplace=True)
df.drop('ID', axis=1, inplace=True)
df.drop('Recording', axis=1, inplace=True)
numberOfAtributtes = len(df.columns)
print(numberOfAtributtes)

mms = MinMaxScaler()
df_norm = mms.fit_transform(df)
clf = SVC()
scores = model_selection.cross_val_score(clf, df_norm, y,
                                         cv=5, scoring='accuracy', n_jobs=-1)
print(scores.mean())

ind = ['poly', 0.3690320297276768, 1.9084110197644817, 0.1053757953826651,
       8.515094980694283]
print(SVCParametersFitness(y, df, numberOfAtributtes, ind))

# # Tworzenie osobnika w reprezentacji binarnej
# def individual(icls):
#     if realRepresentation == False:
#         genome = list()
#         for x in range(0, 40):
#             genome.append(random.randint(0, 1))
#         return icls(genome)
#     else:
#         genome = list()
#         genome.append(random.uniform(-10, 10))
#         genome.append(random.uniform(-10, 10))
#         return icls(genome)
#
#
# def decodeInd(individual):
#     binary_chain_x1 = ''.join(map(str, individual[:20]))
#     binary_chain_x2 = ''.join(map(str, individual[20:]))
#     length_1 = len(binary_chain_x1)
#     length_2 = len(binary_chain_x2)
#     decode_1 = -10 + int(binary_chain_x1, 2) * (10 - (-10)) / (pow(2, length_1) - 1)
#     decode_2 = -10 + int(binary_chain_x2, 2) * (10 - (-10)) / (pow(2, length_2) - 1)
#     return [decode_1, decode_2]


# def fitnessFunction(individual):
#     # Wybrać odpowiednie dla reprezentacji, decode tylko w binarnej
#
#     if realRepresentation == False:
#         ind = decodeInd(individual)
#     else:
#         ind = individual
#     result = (ind[0] + 2 * ind[1] - 7) ** 2 + (2 * ind[0] + ind[1] - 5) ** 2
#     return result,


def plot(generations, val_array, title):
    plt.title(title)
    plt.plot([i for i in range(generations)], val_array, "o")
    plt.show()


if __name__ == '__main__':

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    start = time.time()

    # Wybrać odpowiednie
    if minimum == True:
        creator.create("Individual", list, fitness=creator.FitnessMin)
    else:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # toolbox.register('individual', individual, creator.Individual)
    toolbox.register('individual', SVCParameters, numberOfAtributtes, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("evaluate", fitnessFunction)
    toolbox.register("evaluate", SVCParametersFitness, y, df, numberOfAtributtes)

    # Wybrać selekcje, reszta w komentarzu
    toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("select", tools.selRandom)
    # toolbox.register("select", tools.selBest)
    # toolbox.register("select", tools.selWorst)
    # toolbox.register("select", tools.selRoulette)

    # Wybrać krzyżowanie, reszta w komentarzu
    # toolbox.register("mate", tools.cxOnePoint)
    # toolbox.register("mate", tools.cxUniform, indpb=2)
    # toolbox.register("mate", tools.cxTwoPoint)

    # Krzyżowania dla rzeczywistych
    toolbox.register("mate", arithmetic_crossover, p=0.5)
    # toolbox.register("mate", linear_crossover, p=0.5)
    # toolbox.register("mate", average_crossover, p=0.6)
    # toolbox.register("mate", blend_crossover_alpha, p=0.5, alpha=0.2)
    # toolbox.register("mate", blend_crossover_alpha_beta, p=0.5, alpha=0.2, beta=0.75)

    # Wybrać mutacje, reszta w komentarzu
    # Gaussian używać tylko w reprezentacji rzeczywistej!!!
    # toolbox.register("mutate", tools.mutGaussian, mu=5, sigma=10, indpb=1)
    # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2)
    # toolbox.register("mutate", tools.mutFlipBit, indpb=2)
    # toolbox.register("mutate", tools.mutUniformInt, low=-10, up=10, indpb=2)
    toolbox.register("mutate", mutationSVC)

    sizePopulation = 100
    probabilityMutation = 0.2
    probabilityCrossover = 0.8
    numberIteration = 100

    pop = toolbox.population(n=sizePopulation)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    std_array, avg_array, fit_array, gen_array = [], [], [], []

    g = 0
    while g < numberIteration:
        g = g + 1
        print("-- Generation %i --" % g)

        offspring = toolbox.select(pop, len(pop))

        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # if random.random() < probabilityCrossover:
            #     toolbox.mate(child1, child2)
            #     del child1.fitness.values
            #     del child2.fitness.values
            pass

        for mutant in offspring:
            if random.random() < probabilityMutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # test_fitness = toolbox.evaluate([8, 3, 5, 6, 2])
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # print(" Evaluated %i individuals" % len(invalid_ind))
        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        # print(" Min %s" % min(fits))
        # print(" Max %s" % max(fits))
        # print(" Avg %s" % mean)
        # print(" Std %s" % std)
        best_ind = tools.selBest(pop, 1)[0]

        std_array.append(std)
        avg_array.append(mean)
        fit_array.append(best_ind.fitness.values[0])
        gen_array.append(g)

        # Zakomentować jak jest rzeczywista reprezentacja

        # if realRepresentation == False:
        #     x1, x2 = decodeInd(best_ind)
        # else:
        # x1, x2 = best_ind
        end = time.time()
        # print("Computing time:", str(round((end - start), 2)), "s")
        # print(f"Best individual is {x1}, {x2}, fitness: %s" % best_ind.fitness.values)
        # print(f"f({round(x1, 2)},{round(x2, 2)}) = ", round(best_ind.fitness.values[0], 2))
        # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    print("-- End of (successful) evolution --")

    plot(g, std_array, "Odchylenie standardowe w kolejnej iteracji")
    plot(g, avg_array, "Średnia w kolejnej iteracji")
    plot(g, fit_array, "Funkcja celu w kolejnej iteracji")
