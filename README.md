# Pick up and delivery problem with time windows
University project intended to develop several metaheuristic approaches for solving the pick up and delivery problem with time windows.

## 1. Problem
The pick up and delivery problem with time windows consists of a series of calls which have to be served by a set of vehicles.
An instance of the problem consists of a set of vehicles which has to serve a set of calls.
* Vehicles have a home node, starting time and capacity. For every vehicle, a list of compatible calls is specified.
* Calls consist of an origin node, a destination node, size, cost of not transporting, lower and upper bound time windows for pickup and delivery.
* Topology of the problem is defined by a matrix of costs and times to travel from a node to another for every vehicle.

Five example instances are provided in text files. Format is specified inside those files.
Data about problem instances is loaded using a function provided by [Ramin Hasibi](https://github.com/RaminHasibi/pdp_utils), TA for the course.

## 2. Solution representation
A solution for the problem is represented using a list, encoding which calls are served by each vehicle. Each call appears twice in the list, the
first occurence encodes pickup and the second the delivery. Calls served by a vehicle are delimited by a 0 symbol at the end. An additional "dummy"
vehicle is added to encode not transported calls.

An example solution for the instance `Call_7_Vehicle_3` could be: `[3, 3, 0, 7, 1, 7, 1, 0, 5, 5, 6, 6, 0, 4, 4, 2, 2]`.
Call 3 is served by the first vehicle, calls 7 and 1 are served by the second, calls 5 and 6 are served by the third and calls 4 and 2 are not transported.

A solution is considered to be feasible if:
* For every vehicle, all assigned calls are compatible with it.
* For every vehicle, no call violates its limitations in terms of capacity.
* For every call, its position inside a vehicle does not violate any of the time limitations.

Solutions are checked for feasibility and cost using functions provided by [Ramin Hasibi](https://github.com/RaminHasibi/pdp_utils).

## 3. Metaheuristic algorithms
Three algorithms have been implemented: local search, simulated annealing and a general adaptative metaheuristic. For every algorithm, a series of operators have been implemented (some of them share operators). Each operator has a probability of execution associated with it. A random number is generated and, depending on its value, an operator is applied to the current solution. All algorithms are receive an initial solution with its computed cost, an instance of the corresponding problem and the maximum number of iterations. Depending on the algorithm, other parameters may be required:
* `local_search(init, init_cost, max_iter, probA, probB, probC, problem)`
  * `probA`, `probB` and `probC` are the probabilities associated with each operator.
* `simulated_annealing(init, init_cost, max_iter, probA, probB, probC, t_ini, alpha, problem)`
  * `probA`, `probB` and `probC` are the probabilities associated with each operator. `t_ini` and `alpha` are the initial temperature and the alpha parameter respectively.
* `general_adaptative_metaheuristic(init, init_cost, max_iter, update, beta, problem)`
  * `update` parameter encodes the number of iterations after which probabilities for each operator will be updated. `beta` is used when computing the new probabilities.
