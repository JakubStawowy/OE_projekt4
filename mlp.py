# GaussianProcessClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

import random

from sklearn.neural_network import MLPClassifier


def MLPParametersFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    estimator = MLPClassifier(activation=individual[0], solver=individual[1], alpha=individual[2],
                              max_iter=individual[3], random_state=101)
    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
    predicted = estimator.predict(df_norm[test])
    expected = y[test]
    tn, fp, fn, tp = metrics.confusion_matrix(expected, predicted).ravel()
    result = (tp + tn) / (
            tp + fp + tn + fn)  # w oparciu o macierze pomyłekhttps: // www.dataschool.io / simple - guide - to - confusion - matrixterminology /
    resultSum = resultSum + result  # zbieramy wyniki z poszczególnych etapów walidacji krzyżowej
    return resultSum / split,


def MLPParametersFeatures(numberFeatures, icls):
    genome = list()
    # activation
    activation = ["identity", 'logistic', 'tanh', 'relu']
    genome.append(activation[random.randint(0, 3)])
    # solver
    solver = ["lbfgs", "sgd", "adam"]
    genome.append(solver[random.randint(0, 2)])
    # alpha
    genome.append(random.uniform(0.0001, 0.01))
    # max_iter
    # genome.append(random.randint(100, 300))
    genome.append(1000)
    for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)


def MLPParametersFeatureFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []  # lista cech do usuniecia
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i - numberOfAtributtes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)

    estimator = MLPClassifier(activation=individual[0], solver=individual[1], alpha=individual[2],
                              max_iter=individual[3], random_state=101)

    resultSum = 0
    for train, test in cv.split(df_norm, y):
        estimator.fit(df_norm[train], y[train])
    predicted = estimator.predict(df_norm[test])
    expected = y[test]
    tn, fp, fn, tp = metrics.confusion_matrix(expected,
                                              predicted).ravel()
    result = (tp + tn) / (
            tp + fp + tn + fn)  # w oparciu o macierze pomyłek https: // www.dataschool.io / simple - guide - to - confusion - matrixterminology /
    resultSum = resultSum + result  # zbieramy wyniki z poszczególnych etapów walidacji krzyżowej
    return resultSum / split,


def mutationMLP(individual):
    numberParamer = random.randint(0, len(individual) - 1)
    if numberParamer == 0:
        # activation
        activation = ["identity", 'logistic', 'tanh', 'relu']
        individual[0] = activation[random.randint(0, 3)]
    elif numberParamer == 1:
        # solver
        solver = ["lbfgs", "sgd", "adam"]
        individual[1] = solver[random.randint(0, 2)]
    elif numberParamer == 2:
        # alpha
        individual[2] = random.uniform(0.0001, 0.01)
    elif numberParamer == 3:
        # max_iter
        # individual[3] = random.randint(100, 300)
        individual[3] = 1000
    else:  # genetyczna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0
