
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from ada_boost import ABParametersFitness, ABParametersFeatures, mutationAB
from algorithm import algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from decision_tree import DTParametersFeatures, DTParametersFitness, mutationDT
from gaussian import GAUSSParametersFeatures, mutationGAUSS, GAUSSParametersFitness
from mlp import MLPParametersFeatures, mutationMLP, MLPParametersFitness
from random_forest import RFParametersFeatures, mutationRF, RFParametersFitness
from svc import SVCParametersFitness, mutationSVC, SVCParametersFeatures

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Konfiguracja
# classifier = "SVC"
classifier = "DT"
# classifier = "GAUSS"
# classifier = "RF"
# classifier = "AB"
# classifier = "MLP"

# is_selection = True
is_selection = False

# file = "heart.csv"
file = "data.csv"

realRepresentation = True
minimum = False

sizePopulation = 10
probabilityMutation = 0.2
probabilityCrossover = 0.8
numberIteration = 100

def plot(generations, val_array, title):
    plt.title(title)
    plt.plot([i for i in range(generations)], val_array, "o")
    plt.show()


if __name__ == '__main__':

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

    if classifier == 'SVC':
        clf = SVC()
    elif classifier == 'DT':
        clf = DecisionTreeClassifier()
    elif classifier == 'GAUSS':
        clf = GaussianProcessClassifier()
    elif classifier == 'GAUSS':
        clf = GaussianProcessClassifier()
    elif classifier == 'RF':
        clf = RandomForestClassifier()
    elif classifier == 'AB':
        clf = AdaBoostClassifier()
    elif classifier == 'MLP':
        clf = MLPClassifier()
    else:
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

    # Wybrać odpowiednie
    if minimum:
        creator.create("Individual", list, fitness=creator.FitnessMin)
    else:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    if classifier == 'SVC':
        toolbox.register('individual', SVCParametersFeatures, numberOfAtributtes, creator.Individual)
        toolbox.register("evaluate", SVCParametersFitness, y, df, numberOfAtributtes)
        toolbox.register("mutate", mutationSVC)
    elif classifier == 'DT':
        toolbox.register('individual', DTParametersFeatures, numberOfAtributtes, creator.Individual)
        toolbox.register("evaluate", DTParametersFitness, y, df, numberOfAtributtes)
        toolbox.register("mutate", mutationDT)
    elif classifier == 'GAUSS':
        toolbox.register('individual', GAUSSParametersFeatures, numberOfAtributtes, creator.Individual)
        toolbox.register("evaluate", GAUSSParametersFitness, y, df, numberOfAtributtes)
        toolbox.register("mutate", mutationGAUSS)
    elif classifier == 'RF':
        toolbox.register('individual', RFParametersFeatures, numberOfAtributtes, creator.Individual)
        toolbox.register("evaluate", RFParametersFitness, y, df, numberOfAtributtes)
        toolbox.register("mutate", mutationRF)
    elif classifier == 'AB':
        toolbox.register('individual', ABParametersFeatures, numberOfAtributtes, creator.Individual)
        toolbox.register("evaluate", ABParametersFitness, y, df, numberOfAtributtes)
        toolbox.register("mutate", mutationAB)
    elif classifier == 'MLP':
        toolbox.register('individual', MLPParametersFeatures, numberOfAtributtes, creator.Individual)
        toolbox.register("evaluate", MLPParametersFitness, y, df, numberOfAtributtes)
        toolbox.register("mutate", mutationMLP)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxTwoPoint)

    pop = toolbox.population(n=sizePopulation)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    std_array, avg_array, fit_array, gen_array, best_ind = algorithm(numberIteration, probabilityMutation,
                                                                     probabilityCrossover, toolbox, pop)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    if classifier == 'SVC':
        clf = SVC(kernel=best_ind[0], C=best_ind[1], degree=best_ind[2], gamma=best_ind[3],
                  coef0=best_ind[4], random_state=101)
    elif classifier == 'DT':
        clf = DecisionTreeClassifier(criterion=best_ind[0], splitter=best_ind[1], max_depth=best_ind[2],
                                     min_samples_split=best_ind[3], max_features=best_ind[4], random_state=101)
    elif classifier == 'GAUSS':
        clf = GaussianProcessClassifier(kernel=best_ind[0], n_restarts_optimizer=best_ind[1],
                                        max_iter_predict=best_ind[2], random_state=101)
    elif classifier == 'RF':
        clf = RandomForestClassifier(n_estimators=best_ind[0], criterion=best_ind[1], max_depth=best_ind[2],
                                     max_features=best_ind[3], random_state=101)
    elif classifier == 'AB':
        clf = AdaBoostClassifier(n_estimators=best_ind[0], learning_rate=best_ind[1], algorithm=best_ind[2],
                                 random_state=101)
    elif classifier == 'MLP':
        clf = MLPClassifier(activation=best_ind[0], solver=best_ind[1], alpha=best_ind[2],
                            max_iter=best_ind[3], random_state=101)

    scores = model_selection.cross_val_score(clf, df_norm, y, cv=5, scoring='accuracy', n_jobs=-1)

    if is_selection:
        print("Accuracy after optimalisation with selection ")
    print(f"Accuracy after optimalisation: {scores.mean() * 100}%")

    plot(numberIteration, std_array, "Odchylenie standardowe w kolejnej iteracji")
    plot(numberIteration, avg_array, "Średnia w kolejnej iteracji")
    plot(numberIteration, fit_array, "Funkcja celu w kolejnej iteracji")
