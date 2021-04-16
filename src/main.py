from pdp_utils import load_problem, feasibility_check, cost_function
import numpy as np
import algorithms
import os
import time

if __name__ == '__main__':
    print("Loading datasets...")
    datasets = os.listdir('./datasets')
    print("Datasets loaded: ", datasets)

    parameters_gam = [10000, 1500, 100, 0.35]
    parameters = parameters_gam
    
    file_name = input("\nFile name: ")
    test_r = input("Test(y/n): ")
    test = True if test_r == "y" else False
    if test:
        notes = input("Test notes: ")
        notes = "No notes" if notes == "-" else notes
        num_iters = int(input("Number of rounds: "))

        print("\nRunning test...")
    else:
        num_iters = 10
        print("\nRunning full simulation...")
    
    results = {}
    all_runtimes = []

    tg_1 = time.time()
    for dataset in datasets:
        prob = load_problem('./datasets/' + dataset)
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
            best_solution, cheapest_cost, probabilities = algorithms.general_adaptative_metaheuristic(init, init_cost, parameters_gam[0], parameters_gam[1], parameters_gam[2], parameters_gam[3], prob)
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
    with open(file_name, 'w') as f:
        if test:
            print("--- TEST --- \n", file = f)
            print("NOTES: ", notes, "\n", file = f)
        else:
            print("--- RESULTS --- \n", file = f)

        print("Parameters: ", parameters, "\n", file = f)
        for dataset in datasets:
            print(dataset, ": ", [results[dataset][i] for i in range(len(results[dataset])) if i > 0], file = f)
            print("     Best found solution: ", results[dataset][0], "\n", file = f)

            mean_sum_runtimes += results[dataset][4]
        print("Total runtime: ", (tg_2 - tg_1)/60, "minutes", file = f)
        print("Mean total runtime: ", mean_sum_runtimes, file = f)