from pdp_utils import load_problem, feasibility_check, cost_function
import numpy as np
import algorithms
import os
import time
import random

'''
This function was used for testing purposes

def time_limit(n_calls):
    if n_calls == 7:
        return 10
    elif n_calls == 18:
        return 53
    elif n_calls == 35:
        return 97
    elif n_calls == 80:
        return 155
    elif n_calls == 130:
        return 273
    else:
        return float('inf')
'''

if __name__ == '__main__':
    print("Loading datasets...")
    datasets = os.listdir('./datasets')
    print("Datasets loaded: ", datasets)

    parameters_gam = [-1, -1, 1600, 250, 0.05]
    parameters = parameters_gam
    
    file_name = './output/' + input("\nFile name: ") + '.txt'
    notes = input("Notes: ")
    notes = "No notes" if notes == "-" else notes
    num_iters = int(input("Number of rounds: "))
    plot_weights = False
    if num_iters == 1:
        plot_r = input("Plot weights(y/n): ")
        plot_weights = True if plot_r == "y" else False
        

    print("\nRunning test...")

    results = {}
    all_runtimes = []

    tg_1 = time.time()
    for dataset in datasets:
        prob = load_problem('./datasets/' + dataset)
        parameters_gam[1] = int(input("Total running time: "))
        init = [0]*prob['n_vehicles']
        for i in range(1, prob['n_calls'] + 1):
            init.append(i)
            init.append(i)

        init_cost = cost_function(init, prob)
        runtimes = []
        costs = []
        solutions = []

        print("Current dataset: ", dataset)
        for i in range(num_iters):
            t_ini = time.time()
            best_solution, cheapest_cost, history_probas = algorithms.general_adaptative_metaheuristic(init, init_cost, parameters_gam[0], parameters_gam[1], parameters_gam[2], parameters_gam[3], parameters_gam[4], prob)
            t_end = time.time()

            feasible, _ = feasibility_check(best_solution, prob)
            if (cost_function(best_solution, prob) != cheapest_cost) or not feasible:
                print("WARNING! EITHER CHEAPEST COST DOES NOT MATCH COST FUNCTION VALUE OR THE SOLUTION COMPUTED IS NOT FEASIBLE: ", dataset)
                exit()

            runtimes.append(t_end - t_ini)
            costs.append(cheapest_cost)
            solutions.append(best_solution)

            print("     Iteration: ", i + 1,"\n     Iter. duration: ", t_end - t_ini, " seconds","\n     Total time elapsed: ", (time.time() - tg_1)/60, " minutes", "\n     Improvement: ", 100*(init_cost - cheapest_cost)/init_cost, "% \n")

        cheapest_cost = min(costs)
        best_solution = solutions[costs.index(cheapest_cost)]

        all_runtimes.append(runtimes)
        results[dataset] = [best_solution, np.mean(costs), 100*(init_cost - np.mean(costs))/init_cost, cheapest_cost, 100*(init_cost - cheapest_cost)/init_cost, np.mean(runtimes)]

        if plot_weights:
            name = dataset[:-4]
            with open('./weights_output/' + name + '_probas.csv', 'w') as f:
                print("swap,one_reinsert,costly_reinsert,k_reinsert,three_exchange", file = f)
                for i in range(len(history_probas)):
                    print(history_probas[i][0], ",", history_probas[i][1], ",", history_probas[i][2], ",", history_probas[i][3], ",", history_probas[i][4], file = f)

    tg_2 = time.time()

    mean_sum_runtimes = 0 
    print("Writing to file...")
    with open(file_name, 'w') as f:
        print("--- RESULTS --- \n", file = f)
        print("NOTES: ", notes, "\n", file = f)

        print("Parameters: ", parameters, "\n", file = f)
        for dataset in datasets:
            print(dataset, ": ", [results[dataset][i] for i in range(len(results[dataset])) if i > 0], file = f)
            print("     Best found solution: ", results[dataset][0], "\n", file = f)

            mean_sum_runtimes += results[dataset][5]
        print("Total runtime: ", (tg_2 - tg_1)/60, "minutes", file = f)
        print("Mean total runtime: ", mean_sum_runtimes, file = f)