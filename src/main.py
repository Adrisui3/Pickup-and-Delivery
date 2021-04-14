from pdp_utils import load_problem, feasibility_check, cost_function
import numpy as np
import algorithms
import os
import time

if __name__ == '__main__':
    datasets = os.listdir('../datasets')
    results = {}

    parameters_ls = [0.40, 0.25, 0.35]
    parameters_sa = [0.34, 0.33, 0.33, 1000000, 0.98]
    parameters_gam = [10000, 100, 0.35]
    all_runtimes = []

    test = False
    notes = "Test using the general adaptative metaheuristic. Related k-exchanges and smart_one_reinsert. Testing local search at the end and visited solution scoring option. 10K iter, 1 round."
    parameters = parameters_gam
    num_iters = 1

    print("Computing results...")

    tg_1 = time.time()
    for dataset in datasets:
        prob = load_problem('../datasets/' + dataset)
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
            best_solution, cheapest_cost, probabilities = algorithms.general_adaptative_metaheuristic(init, init_cost, parameters_gam[0], parameters_gam[1], parameters_gam[2], prob)
            t_end = time.time()

            if cost_function(best_solution, prob) != cheapest_cost:
                print("WARNING! CHEAPEST COST AND COST FUNCTION VALUES DO NOT MATCH: ", dataset)
                exit()

            runtimes.append(t_end - t_ini)
            costs.append(cheapest_cost)
            solutions.append(best_solution)

            print("     Iteration: ", i + 1,"\n     Iter. duration: ", t_end - t_ini, " seconds","\n     Total time elapsed: ", (time.time() - tg_1)/60, " minutes", "\n     Probabilities: ", probabilities, "\n")

        cheapest_cost = min(costs)
        best_solution = solutions[costs.index(cheapest_cost)]

        all_runtimes.append(runtimes)
        results[dataset] = [best_solution, np.mean(costs), cheapest_cost, 100*(init_cost - cheapest_cost)/init_cost, np.mean(runtimes)]

    tg_2 = time.time()

    print("Checking runtimes...")
    for i in range(num_iters):
        for j in range(num_iters):
            for k in range(num_iters):
                for n in range(num_iters):
                    for m in range(num_iters):
                        current_runtime = all_runtimes[0][i] + all_runtimes[1][j] + all_runtimes[2][k] + all_runtimes[3][n] + all_runtimes[4][m]
                        if current_runtime > 600:
                            print("\nWARNING!: one combination of executions exceeds 600 seconds runtime, ", current_runtime)

    mean_sum_runtimes = 0 
    print("Writing to file...")
    with open('result.txt', 'w') as f:
        if test:
            print("--- TEST --- \n", file = f)
            print("NOTES: ", notes, "\n", file = f)
        else:
            print("--- RESULTS --- \n", file = f)

        print("Parameters: ", parameters, "\n", file = f)
        for dataset in datasets:
            print(dataset, ": ", results[dataset], "\n", file = f)
            mean_sum_runtimes += results[dataset][4]
        print("Total runtime: ", (tg_2 - tg_1)/60, "minutes", "\n", file = f)
        print("Mean total runtime: ", mean_sum_runtimes, file = f)