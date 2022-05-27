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

from algorithm import algorithm
from crossover import arithmetic_crossover, linear_crossover, blend_crossover_alpha, blend_crossover_alpha_beta, \
    average_crossover
from svc import SVCParameters, SVCParametersFitness, mutationSVC, SVCParametersFeatures


def plot(generations, val_array, title):
    plt.title(title)
    plt.plot([i for i in range(generations)], val_array, "o")
    plt.show()


if __name__ == '__main__':

    # Konfiguracja
    realRepresentation = True
    minimum = False
    is_selection = True
    # is_selection = False
    # file = "heart.csv"
    file = "data.csv"

    pd.set_option('display.max_columns', None)
    df = pd.read_csv(file, sep=',')
    if file == 'heart.csv':
        y = df['target']
        df.drop('target', axis=1, inplace=True)
    else:
        y = df['Status']
        df.drop('Status', axis=1, inplace=True)
        df.drop('ID', axis=1, inplace=True)
        df.drop('Recording', axis=1, inplace=True)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)
    clf = SVC()
    scores = model_selection.cross_val_score(clf, df_norm, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"Accuracy for default configuration: {scores.mean() * 100}%")

    if is_selection:
        if file == 'heart.csv':
            df.drop('sex', axis=1, inplace=True)
            df.drop('chol', axis=1, inplace=True)
            df.drop('cp', axis=1, inplace=True)
        else:
            df.drop('MFCC1', axis=1, inplace=True)
            df.drop('Jitter_rel', axis=1, inplace=True)
            df.drop('Delta12', axis=1, inplace=True)

    numberOfAtributtes = len(df.columns)
    print(numberOfAtributtes)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    start = time.time()

    # Wybrać odpowiednie
    if minimum:
        creator.create("Individual", list, fitness=creator.FitnessMin)
    else:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('individual', SVCParametersFeatures, numberOfAtributtes, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", SVCParametersFitness, y, df, numberOfAtributtes)

    # Wybrać selekcje, reszta w komentarzu
    toolbox.register("select", tools.selTournament, tournsize=3)
    # toolbox.register("select", tools.selRandom)
    # toolbox.register("select", tools.selBest)
    # toolbox.register("select", tools.selWorst)
    # toolbox.register("select", tools.selRoulette)

    # Krzyżowania dla rzeczywistych
    toolbox.register("mate", arithmetic_crossover, p=0.5)
    # toolbox.register("mate", linear_crossover, p=0.5)
    # toolbox.register("mate", average_crossover, p=0.6)
    # toolbox.register("mate", blend_crossover_alpha, p=0.5, alpha=0.2)
    # toolbox.register("mate", blend_crossover_alpha_beta, p=0.5, alpha=0.2, beta=0.75)

    toolbox.register("mutate", mutationSVC)

    sizePopulation = 10
    probabilityMutation = 0.2
    probabilityCrossover = 0.8
    numberIteration = 100

    pop = toolbox.population(n=sizePopulation)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    std_array, avg_array, fit_array, gen_array, best_ind = algorithm(numberIteration, probabilityMutation, toolbox, pop)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)
    clf = SVC(kernel=best_ind[0], C=best_ind[1], degree=best_ind[2], gamma=best_ind[3],
              coef0=best_ind[4], random_state=101)

    # clf = DecisionTreeClassifier(min_weight_fraction_leaf=best_ind[1], ccp_alpha=best_ind[2], random_state=101)

    scores = model_selection.cross_val_score(clf, df_norm, y, cv=5, scoring='accuracy', n_jobs=-1)

    if is_selection:
        print("Accuracy after optimalisation with selection ")
    print(f"Accuracy after optimalisation: {scores.mean() * 100}%")

    plot(numberIteration, std_array, "Odchylenie standardowe w kolejnej iteracji")
    plot(numberIteration, avg_array, "Średnia w kolejnej iteracji")
    plot(numberIteration, fit_array, "Funkcja celu w kolejnej iteracji")
