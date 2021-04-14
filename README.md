# Pick up and delivery with time windows
University project intended to develop several metaheuristic approaches for solving the pick up and delivery problem with time windows.

## 1. Problem
The pick up and delivery problem with time windows consists of a series of calls which have to be served by a set of vehicles.
An instance of the problem consists of a set of vehicles which has to serve a set of calls.
* Vehicles have a home node, starting time and capacity. For every vehicle, a list of compatible calls is specified.
* Calls consist of an origin node, a destination node, size, cost of not transporting, lower and upper bound time windows for pickup and delivery.

Topology of the problem is defined by a matrix of costs and times to travel from a node to another for every vehicle.
