import random
from random import randint
from deap import base
from deap import creator
from deap import tools
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from svc import SVCParameters, SVCParametersFitness


def individual(icls):
    genome = list()
    for x in range(0, 40):
        genome.append(randint(0, 1))
    return icls(genome)


def fitnessFunction(individual):
    # tutaj rozkoduj binarnego osobnika! Napisz funkcje decodeInd
    ind = decodeInd(individual)
    result = (ind[0] + 2 * ind[1] - 7) ** 2 + (2 * ind[0] + ind[1] - 5) ** 2
    return result


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('individual', individual, creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitnessFunction)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)

# pd.set_option('display.max_columns', None)
# df = pd.read_csv("data.csv", sep='\t')
# y = df['Status']
# df.drop('Status', axis=1, inplace=True)
# df.drop('ID', axis=1, inplace=True)
# df.drop('Recording', axis=1, inplace=True)
# numberOfAtributtes = len(df.columns)
# print(numberOfAtributtes)
#
# mms = MinMaxScaler()
# df_norm = mms.fit_transform(df)
# clf = SVC()
# scores = model_selection.cross_val_score(clf, df_norm, y,
#                                          cv=5, scoring='accuracy', n_jobs=-1)
# print(scores.mean())
#
# toolbox.register('individual', SVCParameters, numberOfAtributtes, creator.Individual)
# toolbox.register("evaluate", SVCParametersFitness, y, df, numberOfAtributtes)
#
# ind = ['poly', 0.3690320297276768, 1.9084110197644817, 0.1053757953826651,
#        8.515094980694283]
# print(SVCParametersFitness(y, df, numberOfAtributtes, ind))

sizePopulation = 100
probabilityMutation = 0.2
probabilityCrossover = 0.8
numberIteration = 100

pop = toolbox.population(n=sizePopulation)
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

while g < numberIteration:
    g = g + 1
    print("-- Generation %i --" % g)
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))
    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
    # cross two individuals with probability CXPB
    if random.random() < probabilityCrossover:
        toolbox.mate(child1, child2)
    # fitness values of the children
    # must be recalculated later
    del child1.fitness.values
    del child2.fitness.values
    for mutant in offspring:
    # mutate an individual with probability MUTPB
    if random.random() < probabilityMutation:
        toolbox.mutate(mutant)
    del mutant.fitness.values
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    print(" Evaluated %i individuals" % len(invalid_ind))
    pop[:] = offspring
    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5
    print(" Min %s" % min(fits))
    print(" Max %s" % max(fits))
    print(" Avg %s" % mean)
    print(" Std %s" % std)
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind,
                                         best_ind.fitness.values))
#
print("-- End of (successful) evolution --")


def individual(icls):
    genome = list()
    genome.append(random.uniform(-10, 10))
    genome.append(random.uniform(-10, 10))
    return icls(genome)
