import numpy as np
import random
import copy

##### AUXILIARY FUNCTIONS #####
def calls_vehicle(sol, vehicle):
    solution = sol.copy()
    solution.append(0)
    zeros = [i for i, x in enumerate(solution) if x == 0]
    
    lower_bound, upper_bound = 0 if vehicle == 0 else zeros[vehicle - 1] + 1, zeros[vehicle]
    
    return sol[lower_bound:upper_bound]

def split_unique(sol):
    solution = sol.copy()
    solution.append(0)
    zeros = [i for i, x in enumerate(solution) if x == 0]
    
    unique_vehicles = []
    lower_bound = 0
    for i in range(len(zeros)):
        route = solution[lower_bound:zeros[i]]
        if len(route) > 0:
            unique_vehicles.append(list(set(route)))

        lower_bound = zeros[i] + 1

    return unique_vehicles

def sort_cost(e):
    return e[1]

def size_intersection_compatible_two(call_A, call_B, problem):
    vessel_cargo = problem["VesselCargo"]
    n_vehicles = problem["n_vehicles"]
    
    size_inter, size_A, size_B = 0, 0, 0
    for i in range(n_vehicles):
        if vessel_cargo[i, call_A] and vessel_cargo[i, call_B]:
            size_inter += 1
        
        if vessel_cargo[i, call_A]:
            size_A += 1
        
        if vessel_cargo[i, call_B]:
            size_B += 1

    return size_inter, size_A, size_B

def two_relatedness(call_A, call_B, problem):
    cargo = problem["Cargo"]
    call_A = call_A - 1
    call_B = call_B - 1

    #Time relatedness
    lower_bound_A_P, upper_bound_A_P = cargo[call_A, 4], cargo[call_A, 5]
    lower_bound_B_P, upper_bound_B_P = cargo[call_B, 4], cargo[call_B, 5]

    lower_bound_A_D, upper_bound_A_D = cargo[call_A, 6], cargo[call_A, 7]
    lower_bound_B_D, upper_bound_B_D = cargo[call_B, 6], cargo[call_B, 7]

    time = abs(lower_bound_A_P - lower_bound_B_P) + abs(upper_bound_A_P - upper_bound_B_P) + abs(lower_bound_A_D - lower_bound_B_D) + abs(upper_bound_A_D - upper_bound_B_D)

    #Size relatedness
    size_A, size_B = cargo[call_A, 2], cargo[call_B, 2]

    size = abs(size_A - size_B)

    #Compatibility relatedness
    size_inter, size_set_A, size_set_B = size_intersection_compatible_two(call_A, call_B, problem)    
    compatibility_relatedness = 1 - size_inter/(min(size_set_A, size_set_B))

    return (1/3)*time + (1/3)*size + (1/3)*compatibility_relatedness

def size_intersection_compatible_three(call_A, call_B, call_C, problem):
    vessel_cargo = problem["VesselCargo"]
    n_vehicles = problem["n_vehicles"]
    
    size_inter, size_A, size_B, size_C = 0, 0, 0, 0
    for i in range(n_vehicles):
        if vessel_cargo[i, call_A] and vessel_cargo[i, call_B] and vessel_cargo[i, call_C]:
            size_inter += 1
        
        if vessel_cargo[i, call_A]:
            size_A += 1
        
        if vessel_cargo[i, call_B]:
            size_B += 1
        
        if vessel_cargo[i, call_C]:
            size_C += 1
        
    return size_inter, size_A, size_B, size_C

def three_relatedness(call_A, call_B, call_C, problem):
    cargo = problem["Cargo"]
    call_A = call_A - 1
    call_B = call_B - 1
    call_C = call_C - 1

    #Time relatedness
    lower_bound_A_P, upper_bound_A_P = cargo[call_A, 4], cargo[call_A, 5]
    lower_bound_B_P, upper_bound_B_P = cargo[call_B, 4], cargo[call_B, 5]
    lower_bound_C_P, upper_bound_C_P = cargo[call_C, 4], cargo[call_C, 5]

    lower_bound_A_D, upper_bound_A_D = cargo[call_A, 6], cargo[call_A, 7]
    lower_bound_B_D, upper_bound_B_D = cargo[call_B, 6], cargo[call_B, 7]
    lower_bound_C_D, upper_bound_C_D = cargo[call_C, 6], cargo[call_C, 7]

    time = abs(lower_bound_A_P - lower_bound_B_P) + abs(lower_bound_A_P - lower_bound_C_P) + abs(lower_bound_B_P - lower_bound_C_P)\
         + abs(upper_bound_A_P - upper_bound_B_P) + abs(upper_bound_A_P - upper_bound_C_P) + abs(upper_bound_B_P - upper_bound_C_P)\
         + abs(lower_bound_A_D - lower_bound_B_D) + abs(lower_bound_A_D - lower_bound_C_D) + abs(lower_bound_B_D - lower_bound_C_D)\
         + abs(upper_bound_A_D - upper_bound_B_D) + abs(upper_bound_A_D - upper_bound_C_D) + abs(upper_bound_B_D - upper_bound_C_D)

    #Size relatedness
    size_A, size_B, size_C = cargo[call_A, 2], cargo[call_B, 2], cargo[call_C, 2]

    size = abs(size_A - size_B) + abs(size_A - size_C) + abs(size_B - size_C)

    #Compatibility relatedness
    size_inter, size_set_A, size_set_B, size_set_C = size_intersection_compatible_three(call_A, call_B, call_C, problem)
    compatibility_relatedness = 1 - size_inter/(min(size_set_A, size_set_B, size_set_C))

    return (1/3)*time + (1/3)*size + (1/3)*compatibility_relatedness

def list_length(li):
    return len(li)

def route_cost(vehicle, route, problem):
    if len(route) == 0:
        return 0
    
    Cargo = problem['Cargo']
    TravelCost = problem['TravelCost']
    FirstTravelCost = problem['FirstTravelCost']
    PortCost = problem['PortCost']
    currentVPlan = route.copy()
    currentVPlan = [x - 1 for x in currentVPlan]

    sortRout = np.sort(currentVPlan, kind='mergesort')
    I = np.argsort(currentVPlan, kind='mergesort')
    Indx = np.argsort(I, kind='mergesort')

    PortIndex = Cargo[sortRout, 1].astype(int)
    PortIndex[::2] = Cargo[sortRout[::2], 0]
    PortIndex = PortIndex[Indx] - 1

    Diag = TravelCost[vehicle, PortIndex[:-1], PortIndex[1:]]

    FirstVisitCost = FirstTravelCost[vehicle, int(Cargo[currentVPlan[0], 0] - 1)]
    RouteTravelCost = np.sum(np.hstack((FirstVisitCost, Diag.flatten())))
    CostInPorts = np.sum(PortCost[vehicle, currentVPlan]) / 2

    return RouteTravelCost + CostInPorts

def feasible_vehicle(vehicle, route, problem):
    len_route = len(route)
    
    if len_route == 0:
        return True

    cargo = problem['Cargo']
    travel_time = problem['TravelTime']
    first_travel_time = problem['FirstTravelTime']
    vessel_capacity = problem['VesselCapacity']
    loading_time = problem['LoadingTime']
    unloading_time = problem['UnloadingTime']

    load_size, current_time = 0, 0
    route_aux = route.copy()
    route_aux = [x - 1 for x in route_aux]
    sorted_route = np.sort(route_aux, kind='mergesort')
    I = np.argsort(route_aux, kind='mergesort')
    index = np.argsort(I, kind='mergesort')

    load_size -=cargo[sorted_route, 2]
    load_size[::2] = cargo[sorted_route[::2], 2]
    load_size = load_size[index]
    if np.any(vessel_capacity[vehicle] - np.cumsum(load_size) < 0):
        return False
    
    time_windows = np.zeros((2, len_route))
    time_windows[0] = cargo[sorted_route, 6]
    time_windows[0, ::2] = cargo[sorted_route[::2], 4]
    time_windows[1] = cargo[sorted_route, 7]
    time_windows[1, ::2] = cargo[sorted_route[::2], 5]
    time_windows = time_windows[:, index]

    port_index = cargo[sorted_route, 1].astype(int)
    port_index[::2] = cargo[sorted_route[::2], 0]
    port_index = port_index[index] - 1

    lu_time = unloading_time[vehicle, sorted_route]
    lu_time[::2] = loading_time[vehicle, sorted_route[::2]]
    lu_time = lu_time[index]
    diag = travel_time[vehicle, port_index[:-1], port_index[1:]]
    first_visit_time = first_travel_time[vehicle, int(cargo[route_aux[0], 0] - 1)]
    route_travel_time = np.hstack((first_visit_time, diag.flatten()))

    arrive_time = np.zeros(len_route)
    for i in range(len_route):
        arrive_time[i] = np.max((current_time + route_travel_time[i], time_windows[0, i]))
        if arrive_time[i] > time_windows[1, i]:
            return False
        current_time = arrive_time[i] + lu_time[i]
    
    return True


#####################################################################

##### OPERATORS #####

def swap(sol):
    unique_vehicles = split_unique(sol)
    candidates = []
    elements = []

    if len(unique_vehicles) >= 2:
        candidates = random.sample(range(len(unique_vehicles)), 2)
        elem_1 = random.choice(unique_vehicles[candidates[0]])
        elem_2 = random.choice(unique_vehicles[candidates[1]])

        elements = [elem_1, elem_2]
    else:
        elements = random.sample(unique_vehicles[0], 2)

    neighbor = copy.copy(sol)
    for i in range(len(sol)):
        if sol[i] == elements[0]:
            neighbor[i] = elements[1]
        elif sol[i] == elements[1]:
            neighbor[i] = elements[0]

    return neighbor

def three_exchange(sol):
    unique_vehicles = split_unique(sol)
    elements = []
    candidates = []

    if len(unique_vehicles) >= 3:
        candidates = random.sample(range(len(unique_vehicles)), 3)
        elem_1 = random.choice(unique_vehicles[candidates[0]])
        elem_2 = random.choice(unique_vehicles[candidates[1]])
        elem_3 = random.choice(unique_vehicles[candidates[2]])

        elements = [elem_1, elem_2, elem_3]
    elif len(unique_vehicles) == 2:
        if max(len(unique_vehicles[0]), len(unique_vehicles[1])) == len(unique_vehicles[0]):
            elements = random.sample(unique_vehicles[0], 2)
            elements.append(random.choice(unique_vehicles[1]))
        else:
            elements = random.sample(unique_vehicles[1], 2)
            elements.append(random.choice(unique_vehicles[0]))
    else:
        elements = random.sample(unique_vehicles[0], 3)

    neighbor = []
    for x in sol:
        if x == elements[0]:
            neighbor.append(elements[1])
        elif x == elements[1]:
            neighbor.append(elements[2])
        elif x == elements[2]:
            neighbor.append(elements[0])
        else:
            neighbor.append(x)

    return neighbor

def one_reinsert(sol, problem):
    chosen = random.randint(1, problem['n_calls'])
    aux_sol = [x for x in sol if x != chosen]
    broken_sol = []
    for i in range(problem['n_vehicles'] + 1):
        broken_sol.append(calls_vehicle(aux_sol, i))

    candidate = random.randint(0, problem['n_vehicles'])

    if len(broken_sol[candidate]) > 0:
        pos_1 = random.choice(range(len(broken_sol[candidate])))
        broken_sol[candidate].insert(pos_1, chosen)

        pos_2 = random.choice(range(len(broken_sol[candidate])))
        broken_sol[candidate].insert(pos_2, chosen)
    else:
        broken_sol[candidate].append(chosen)
        broken_sol[candidate].append(chosen)

    neighbor = []
    for i in range(len(broken_sol)):
        if i + 1 <= problem['n_vehicles']:
            broken_sol[i].append(0)

        neighbor += broken_sol[i]

    return neighbor

def related_swap(solution, problem):
    unique_vehicles = split_unique(solution)
    if len(unique_vehicles) >= 2:
        candidates = random.sample(unique_vehicles, 2)
        call_A = random.choice(candidates[0])

        elegible_B = [(x, two_relatedness(call_A, x, problem)) for x in candidates[1]]
        call_B = min(elegible_B, key=sort_cost)[0]
    else:
        call_A = random.randint(1, problem['n_calls'])
        candidates = [(x, two_relatedness(call_A, x, problem)) for x in range(1, problem['n_calls'] + 1) if x != call_A]
        call_B = min(candidates, key=sort_cost)[0]

    neighbor = []
    for x in solution:
        if x == call_A:
            neighbor.append(call_B)
        elif x == call_B:
            neighbor.append(call_A)
        else:
            neighbor.append(x)

    return neighbor

def related_three_exchange(solution, problem):
    unique_vehicles = split_unique(solution)
    if len(unique_vehicles) >= 3:
        candidates = random.sample(unique_vehicles, 3)
        call_A = random.choice(candidates[0])

        elegible_B = [(x, two_relatedness(call_A, x, problem)) for x in candidates[1]]
        call_B = min(elegible_B, key=sort_cost)[0]

        elegible_C = [(x, three_relatedness(call_A, call_B, x, problem)) for x in candidates[2]]
        call_C = min(elegible_C, key=sort_cost)[0]
    elif len(unique_vehicles) == 2:
        elegible_A_B = max(unique_vehicles, key=list_length)
        call_A = random.choice(elegible_A_B)
        
        elegible_A_B.remove(call_A)
        candidates_B = [(x, two_relatedness(call_A, x, problem)) for x in elegible_A_B]
        call_B = min(candidates_B, key=sort_cost)[0]

        elegible_C = min(unique_vehicles, key=list_length)
        candidates_C = [(x, three_relatedness(call_A, call_B, x, problem)) for x in elegible_C]
        call_C = min(candidates_C, key=sort_cost)[0]
    else:
        call_A = random.randint(1, problem['n_calls'])
        candidates = [x for x in range(1, problem['n_calls'] + 1) if x != call_A]

        candidates_relatedness = [(x, two_relatedness(call_A, x, problem)) for x in candidates]
        call_B = min(candidates_relatedness, key=sort_cost)[0]

        candidates.remove(call_B)
        candidates_three_relatedness = [(x, three_relatedness(call_A, call_B, x, problem)) for x in candidates]
        call_C = min(candidates_three_relatedness, key=sort_cost)[0]

    neighbor = []
    for x in solution:
        if x == call_A:
            neighbor.append(call_B)
        elif x == call_B:
            neighbor.append(call_C)
        elif x == call_C:
            neighbor.append(call_A)
        else:
            neighbor.append(x)

    return neighbor

def smart_one_reinsert(solution, problem):
    n_calls = problem['n_calls']
    n_vehicles = problem['n_vehicles']
    vessel_cargo = problem['VesselCargo']

    call = random.randint(1, n_calls)
    
    base_solution = [x for x in solution if x != call]
    zeros = [i for i, x in enumerate(base_solution) if x == 0]
    compatible_vehicles = [i for i in range(n_vehicles) if vessel_cargo[i, call - 1]]

    #neighbors = []
    #costs = []
    for vehicle in compatible_vehicles:
        neighbor = base_solution.copy()
        lower_bound, upper_bound = 0 if vehicle == 0 else zeros[vehicle - 1] + 1, zeros[vehicle]
        route = base_solution[lower_bound:upper_bound]
        best_position = (-1, -1)
        min_cost = float('inf')

        for i in range(len(route)):
            first_insert_route = route.copy()
            first_insert_route.insert(i, call)
            for j in range(i, len(route)):
                second_insert_route = first_insert_route.copy()
                second_insert_route.insert(j, call)

                if not feasible_vehicle(vehicle, second_insert_route, problem):
                    continue
                
                cost = route_cost(vehicle, second_insert_route, problem)
                if cost < min_cost:
                    min_cost = cost
                    best_position = (i, j)
        
        if len(route) > 0 and best_position == (-1, -1):
            continue
        elif len(route) == 0:
            best_position = (0, 0)
        
        neighbor.insert(lower_bound + best_position[0], call)
        neighbor.insert(lower_bound + best_position[1], call)

        return neighbor

        #delta = min_cost - route_cost(vehicle, route, problem)

        #neighbors.append(neighbor)
        #costs.append(delta)

    return solution
    #return neighbors[costs.index(min(costs))] if len(costs) > 0 else solution
