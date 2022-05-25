import math
import random
from functools import cmp_to_key

# Wybrać czy max czy min
GOAL = "MINIMUM"
# GOAL = "MAXIMUM"


def fitnessFunction(individual):
    ind = individual
    result = (ind[0] + 2 * ind[1] - 7) ** 2 + (2 * ind[0] + ind[1] - 5) ** 2
    return result


def ind_comparator(ind_1, ind_2):
    direction = 0
    if GOAL == "MAXIMUM":
        direction = 1
    elif GOAL == "MINIMUM":
        direction = -1

    try:
        return direction * int(math.fabs(fitnessFunction(ind_2)) - math.fabs(fitnessFunction(ind_1)))
    except:
        return None


def arithmetic_crossover(ind_1, ind_2, p):
    if random.random() > p:
        return [ind_1, ind_2]

    k = random.random()
    k_reverse = 1 - k
    x1 = ind_1[0]
    y1 = ind_1[1]

    x2 = ind_2[0]
    y2 = ind_2[1]

    # Witam tutaj moze byc problem
    x1_new = k * x1 + k_reverse * x2
    y1_new = k * y1 + k_reverse * y2

    x2_new = k_reverse * x1 + k * x2
    y2_new = k_reverse * y1 + k * y2

    return [[x1_new, y1_new], [x2_new, y2_new]]


def linear_crossover(ind_1, ind_2, p):
    if random.random() > p:
        return [ind_1, ind_2]

    x1 = ind_1[0]
    y1 = ind_1[1]

    x2 = ind_2[0]
    y2 = ind_2[1]
    z_ind = [(x1 + x2) / 2, (y1 + y2) / 2]
    v_ind = [3 * x1 / 2 - x2 / 2, 3 * y1 / 2 - y2 / 2]
    w_ind = [3 * x2 / 2 - x1 / 2, 3 * y2 / 2 - y1 / 2]

    ind_array = [z_ind, v_ind, w_ind]
    result = []
    for ind in ind_array:
        # new_ind = Individual([Chromosome(representation=ind[0]), Chromosome(representation=ind[1])])
        result.append([ind[0], ind[1]])

    sorted_array = sorted(result, key=cmp_to_key(ind_comparator))

    return sorted_array[:-1]


# Sprawdzić bo we wzorze jest max(y1,xy), ale liczone jest potem jak u nas, trzeba sprawdzić

def blend_crossover_alpha(ind_1, ind_2, p, alpha, start=-10, end=10):
    if random.random() > p:
        return [ind_1, ind_2]

    x1 = ind_1[0]
    y1 = ind_1[1]

    x2 = ind_2[0]
    y2 = ind_2[1]

    d1 = abs(x1 - x2)
    d2 = abs(y1 - y2)

    new_ind_array = []
    while len(new_ind_array) < 2:
        u1 = random.uniform(min(x1, x2) - alpha * d1, max(x1, x2) + alpha * d1)
        u2 = random.uniform(min(y1, y2) - alpha * d2, max(y1, y2) + alpha * d2)

        if u1 >= start and u1 <= end and u2 >= start and u2 <= end:
            new_ind_array.append([u1, u2])

    return new_ind_array


def blend_crossover_alpha_beta(ind_1, ind_2, p, alpha, beta, start=-10, end=10):
    if random.random() > p:
        return [ind_1, ind_2]

    x1 = ind_1[0]
    y1 = ind_1[1]

    x2 = ind_2[0]
    y2 = ind_2[1]

    d1 = abs(x1 - x2)
    d2 = abs(y1 - y2)

    new_ind_array = []
    while len(new_ind_array) < 2:
        u1 = random.uniform(min(x1, x2) - alpha * d1, max(x1, x2) + beta * d1)
        u2 = random.uniform(min(y1, y2) - alpha * d2, max(y1, y2) + beta * d2)

        if u1 >= start and u1 <= end and u2 >= start and u2 <= end:
            new_ind_array.append([u1, u2])

    return new_ind_array


def average_crossover(ind_1, ind_2, p):
    if random.random() > p:
        return [ind_1, ind_2]

    x1 = ind_1[0]
    y1 = ind_1[1]

    x2 = ind_2[0]
    y2 = ind_2[1]

    u1 = (x1 + x2) / 2
    u2 = (y1 + y2) / 2

    return [[u1, u2], [u1, u2]]
