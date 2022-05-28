from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

import random

from sklearn.ensemble import RandomForestClassifier


def RFParametersFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    estimator = RandomForestClassifier(n_estimators=individual[0], criterion=individual[1], max_depth=individual[2],
                                       max_features=individual[3], random_state=101)
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


def RFParametersFeatures(numberFeatures, icls):
    genome = list()
    # n_estimators
    n_estimators = random.randint(100, 150)
    genome.append(n_estimators)
    # criteria
    listCrit = ["gini", "entropy"]
    genome.append(listCrit[random.randint(0, 1)])
    # max_depth
    genome.append(random.randint(1, 100))
    # max_features
    listMaxFeat = ["sqrt", "log2"]
    genome.append(listMaxFeat[random.randint(0, 1)])
    for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)


def RFParametersFeatureFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []  # lista cech do usuniecia
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i - numberOfAtributtes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)

    estimator = RandomForestClassifier(n_estimators=individual[0], criterion=individual[1], max_depth=individual[2],
                                       max_features=individual[3], random_state=101)

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

    # estimator = RandomForestClassifier(n_estimators=individual[0], criterion=individual[1], max_depth=individual[2],
    #                                    max_features=individual[3], random_state=101)


def mutationRF(individual):
    numberParamer = random.randint(0, len(individual) - 1)
    if numberParamer == 0:
        # n_estimators
        n_estimators = random.randint(100, 150)
        individual[0] = n_estimators
    elif numberParamer == 1:
        # criterion
        listCrit = ["gini", "entropy"]
        individual[1] = listCrit[random.randint(0, 1)]
    elif numberParamer == 2:
        # max_depth
        individual[2] = random.randint(1, 100)
    elif numberParamer == 3:
        # max_features
        listMaxFeat = ["sqrt", "log2"]
        individual[3] = listMaxFeat[random.randint(0, 1)]
    else:  # genetyczna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0
