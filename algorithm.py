import random

from deap import tools


def algorithm(numberIteration, probabilityMutation, toolbox, pop):

    std_array, avg_array, fit_array, gen_array = [], [], [], []
    g = 0
    best_ind = []
    while g < numberIteration:

        g = g + 1

        offspring = toolbox.select(pop, len(pop))

        offspring = list(map(toolbox.clone, offspring))

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
        # end = time.time()
        # print("Computing time:", str(round((end - start), 2)), "s")
        # print(f"Best individual is {x1}, {x2}, fitness: %s" % best_ind.fitness.values)
        # print(f"f({round(x1, 2)},{round(x2, 2)}) = ", round(best_ind.fitness.values[0], 2))
        # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    print("-- End of (successful) evolution --")
    print("Best individual:")
    print(f"Kernel: {best_ind[0]} | {best_ind[1]}, {best_ind[2]}, {best_ind[3]}, {best_ind[4]}")
    print(f"Fitness value {best_ind.fitness.values[0]}")
    print("=====================")

    return std_array, avg_array, fit_array, gen_array, best_ind
