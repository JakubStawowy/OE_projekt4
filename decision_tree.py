
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn import metrics

import random

from sklearn.tree import DecisionTreeClassifier

def DTParametersFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)
    mms = MinMaxScaler()
    df_norm = mms.fit_transform(df)

    estimator = DecisionTreeClassifier(criterion=individual[0], splitter=individual[1], max_depth=individual[2],
                                       min_samples_split=individual[3], max_features=individual[4], random_state=101)
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


def DTParametersFeatures(numberFeatures, icls):
    genome = list()
    # criterion
    listCrit = ["gini", "entropy"]
    genome.append(listCrit[random.randint(0, 1)])
    # splitter
    listSplit = ["best", "random"]
    k = listSplit[random.randint(0, 1)]
    genome.append(k)
    # max_depth
    genome.append(random.randint(0, 100))
    # min_samples_split
    genome.append(random.randint(2, 10))
    # max_features
    listMaxFeat = ["auto", "sqrt", "log2"]
    genome.append(listMaxFeat[random.randint(0, 2)])
    for i in range(0, numberFeatures):
        genome.append(random.randint(0, 1))
    return icls(genome)


def DTParametersFeatureFitness(y, df, numberOfAtributtes, individual):
    split = 5
    cv = StratifiedKFold(n_splits=split)

    listColumnsToDrop = []  # lista cech do usuniecia
    for i in range(numberOfAtributtes, len(individual)):
        if individual[i] == 0:  # gdy atrybut ma zero to usuwamy cechę
            listColumnsToDrop.append(i - numberOfAtributtes)
    dfSelectedFeatures = df.drop(df.columns[listColumnsToDrop], axis=1, inplace=False)

    mms = MinMaxScaler()
    df_norm = mms.fit_transform(dfSelectedFeatures)

    estimator = DecisionTreeClassifier(criterion=individual[0], splitter=individual[1], max_depth=individual[2],
                                       min_samples_split=individual[3], max_features=individual[4], random_state=101)

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


def mutationDT(individual):
    numberParamer = random.randint(0, len(individual) - 1)
    if numberParamer == 0:
        # criterion
        listCrit = ["gini", "entropy"]
        individual[0] = listCrit[random.randint(0, 1)]
    elif numberParamer == 1:
        # splitter
        listSplit = ["best", "random"]
        individual[1] = listSplit[random.randint(0, 1)]
    elif numberParamer == 2:
        # max_depth
        individual[2] = random.randint(0, 100)
    elif numberParamer == 3:
        # min_samples_split
        individual[3] = random.randint(2, 10)
    elif numberParamer == 4:
        # max_features
        listMaxFeat = ["auto", "sqrt", "log2"]
        individual[4] = listMaxFeat[random.randint(0, 2)]
    else:  # genetyczna selekcja cech
        if individual[numberParamer] == 0:
            individual[numberParamer] = 1
        else:
            individual[numberParamer] = 0
