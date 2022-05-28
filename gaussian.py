from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.gaussian_process.kernels import RBF

import random

from sklearn.gaussian_process import GaussianProcessClassifier


def GAUSSParametersFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    estimator = GaussianProcessClassifier(kernel=individual[0], n_restarts_optimizer=individual[1],
                                          max_iter_predict=individual[2], random_state=101)
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


def GAUSSParametersFeatures(numberFeatures, icls):
    genome = list()
    # kernel
    kernel = random.random() * RBF()
    genome.append(kernel)
    # n_restarts_optimizer
    k = random.randint(0, 20)
    genome.append(k)
    # max_iter_predict
    genome.append(random.randint(100, 300))
    for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)


def GAUSSParametersFeatureFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []  # lista cech do usuniecia
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i - numberOfAtributtes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)

    estimator = GaussianProcessClassifier(kernel=individual[0], n_restarts_optimizer=individual[1],
                                          max_iter_predict=individual[2], random_state=101)

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


def mutationGAUSS(individual):
    numberParamer = random.randint(0, len(individual) - 1)
    if numberParamer == 0:
        # kernel
        kernel = random.random() * RBF()
        individual[0] = kernel
    elif numberParamer == 1:
        # n_restarts_optimizer
        k = random.randint(0, 20)
        individual[1] = k
    elif numberParamer == 2:
        # max_iter_predict
        individual[2] = random.randint(100, 300)
    else:  # genetyczna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0
