from pdp_utils import load_problem, feasibility_check, cost_function
import numpy as np
import operators
import random


def local_search(init, init_cost, max_iter, probA, probB, probC, problem):
    current = init
    best_solution = init
    cheapest_cost = init_cost

    for _ in range(max_iter):
        ran = random.uniform(0, 1)
        if ran < probA:
            current = operators.swap(best_solution)
        elif ran < probA + probB:
            current = operators.three_exchange(best_solution)
        else:
            current = operators.one_reinsert(best_solution, problem)

        feasible, _ = feasibility_check(current, problem)
        current_cost = cost_function(current, problem)

        if feasible and current_cost < cheapest_cost:
            best_solution = current
            cheapest_cost = current_cost

    return best_solution, cheapest_cost

def local_search_memoization(init, init_cost, feasible_solutions, infeasible_solutions, max_iter, probA, probB, probC, problem):
    current = init
    best_solution = init
    cheapest_cost = init_cost

    for _ in range(max_iter):
        ran = random.uniform(0, 1)
        if ran < probA:
            current = operators.swap(best_solution)
        elif ran < probA + probB:
            current = operators.three_exchange(best_solution)
        else:
            current = operators.one_reinsert(best_solution, problem)
        
        if tuple(current) in feasible_solutions:
            feasible = True
            current_cost = feasible_solutions[tuple(current)]
        elif tuple(current) in infeasible_solutions:
            feasible = False
        else:
            feasible, _ = feasibility_check(current, problem)
            current_cost = cost_function(current, problem)

        if feasible:
            if tuple(current) not in feasible_solutions:
                feasible_solutions[tuple(current)] = current_cost

            if current_cost < cheapest_cost:
                best_solution = current
                cheapest_cost = current_cost
        else:
            if tuple(current) not in infeasible_solutions:
                infeasible_solutions.add(tuple(current))

    return best_solution, cheapest_cost

def simulated_annealing(init, init_cost, max_iter, probA, probB, probC, t_ini, alpha, problem):
    incumbent = init
    best_solution = init
    cheapest_cost = init_cost
    temp = t_ini

    for _ in range(max_iter):
        ran = random.uniform(0, 1)
        if ran < probA:
            current = operators.related_swap(incumbent, problem)
        elif ran < probA + probB:
            current = operators.related_three_exchange(incumbent, problem)
        else:
            current = operators.smart_one_reinsert(incumbent, problem)

        current_cost = cost_function(current, problem)
        incumbent_cost = cost_function(incumbent, problem)
        delta = current_cost - incumbent_cost

        feasible, _ = feasibility_check(current, problem)

        if feasible:
            if delta < 0:
                incumbent = current
                if current_cost < cheapest_cost:
                    best_solution = incumbent
                    cheapest_cost = current_cost
            elif random.uniform(0.01, 0.99) < np.exp(-delta/temp):
                incumbent = current

        temp = temp*alpha

    return best_solution, cheapest_cost

def general_adaptative_metaheuristic(init, init_cost, max_iter, max_iter_ls, update, beta, problem):
    incumbent = init
    best_solution = init
    cheapest_cost = init_cost
    probA, probB, probC = 1/3, 1/3, 1/3
    scores, times = [0, 0, 0], [0, 0, 0]
    current_op = -1
    feasible_solutions = dict()
    infeasible_solutions = set()
    feasible_solutions[tuple(init)] = init_cost

    for i in range(max_iter):
        ran = random.uniform(0, 1)
        if ran < probA:
            current = operators.related_swap(incumbent, problem)
            current_op = 0
        elif ran < probA + probB:
            current = operators.related_three_exchange(incumbent, problem)
            current_op = 1
        else:
            current = operators.smart_one_reinsert(incumbent, problem)
            current_op = 2

        times[current_op] += 1

        if tuple(current) in feasible_solutions:
            feasible = True
        elif tuple(current) in infeasible_solutions:
            feasible = False
        else:
            feasible, _ = feasibility_check(current, problem)

        if feasible:
            if tuple(current) not in feasible_solutions:
                feasible_solutions[tuple(current)] = cost_function(current, problem)
                scores[current_op] += 1
            
            delta = feasible_solutions[tuple(current)] - feasible_solutions[tuple(incumbent)]
            if delta < 0:
                incumbent = current
                scores[current_op] += 2
                if feasible_solutions[tuple(incumbent)] < cheapest_cost:
                    best_solution = incumbent
                    cheapest_cost = feasible_solutions[tuple(incumbent)]
                    scores[current_op] += 4
            elif feasible_solutions[tuple(current)] < cheapest_cost + 0.2*((max_iter - i)/max_iter)*cheapest_cost:
                incumbent = current
        else:
            if tuple(current) not in infeasible_solutions:
                infeasible_solutions.add(tuple(current))

        if i % update == 0:
            probA = probA * (1 - beta) + beta*(scores[0]/times[0]) if times[0] > 0 else probA
            probB = probB * (1 - beta) + beta*(scores[1]/times[1]) if times[1] > 0 else probB
            probC = probC * (1 - beta) + beta*(scores[2]/times[2]) if times[2] > 0 else probC

            suma = probA + probB + probC
            probA, probB, probC = probA/suma, probB/suma, probC/suma
            
            #We make sure the algorithm does not kill any operator
            if probA < 0.1 or probB < 0.1 or probC < 0.1:
                deltaA, deltaB, deltaC = 0, 0, 0
                if probA < 0.1:
                    deltaA = 0.1 - probA
                    probA = 0.1

                if probB < 0.1:
                    deltaB = 0.1 - probB
                    probB = 0.1

                if probC < 0.1:
                    deltaC = 0.1 - probC
                    probC = 0.1
                
                max_prob = max(probA , probB, probC)
                if max_prob == probA:
                    probA -= deltaA + deltaB + deltaC
                elif max_prob == probB:
                    probB -= deltaA + deltaB + deltaC
                else:
                    probC -= deltaA + deltaB + deltaC
                
                suma = probA + probB + probC
                if suma != 1.0:
                    probA, probB, probC = probA/suma, probB/suma, probC/suma
    
    #Quick local search in order to find local optima if possible
    best_solution, cheapest_cost = local_search_memoization(best_solution, cheapest_cost, feasible_solutions, infeasible_solutions, max_iter_ls, 1/3, 1/3, 1/3, problem)

    return best_solution, cheapest_cost, [probA, probB, probC]