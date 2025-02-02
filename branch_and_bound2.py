import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from collections import deque
import problems as pr

# Define global variables
isMax = None  # Max / Min sense
DEBUG_MODE = True  # Debug mode
nodes = 0  # Number of nodes explored
lower_bound = -np.inf  # Lower bound of the problem
upper_bound = np.inf  # Upper bound of the problem

# Calculate if a value is very close to an integer value
def is_nearly_integer(value, tolerance=1e-6):
    return abs(value - round(value)) <= tolerance

# a class 'Node' that holds information of a node
class Node:
    def __init__(self, ub, lb, depth, vbasis, cbasis, branching_var, label=""):
        self.ub = ub
        self.lb = lb
        self.depth = depth
        self.vbasis = vbasis
        self.cbasis = cbasis
        self.branching_var = branching_var
        self.label = label

# print debugging info
def debug_print(node: Node = None, x_obj=None, sol_status=None):
    print("\n\n-----------------  DEBUG OUTPUT  -----------------\n\n")
    print(f"UB:{upper_bound}")
    print(f"LB:{lower_bound}")
    if node is not None:
        print(f"Branching Var: {node.branching_var}")
    if node is not None:
        print(f"Child: {node.label}")
    if node is not None:
        print(f"Depth: {node.depth}")
    if x_obj is not None:
        print(f"Simplex Objective: {x_obj}")
    if sol_status is not None:
        print(f"Solution status: {sol_status}")

    print("\n\n--------------------------------------------------\n\n")


# Select branching variable based on heuristics
def select_branching_variable(x_candidate, integer_var, heuristic="max_fractional"):
    """
    Selects the variable for branching using different heuristics.
    """
    fractional_vars = [(idx, abs(x_candidate[idx] - round(x_candidate[idx]))) 
                       for idx, is_int in enumerate(integer_var) if is_int and not is_nearly_integer(x_candidate[idx])]

    if not fractional_vars:
        return None  # All integer variables are already integer

    if heuristic == "max_fractional":
        return max(fractional_vars, key=lambda x: x[1])[0]
    elif heuristic == "min_fractional":
        return min(fractional_vars, key=lambda x: x[1])[0]
    elif heuristic == "random":
        return np.random.choice([var[0] for var in fractional_vars])
    else:
        return max(fractional_vars, key=lambda x: x[1])[0]  # Default: max fractional

# Select branching order based on heuristics
def select_branching_order(x_value, heuristic="round_down_first"):
    """
    Selects the branching order.
    """
    lower_bound = np.floor(x_value)
    upper_bound = np.ceil(x_value)

    if heuristic == "round_down_first":
        return [lower_bound, upper_bound]
    elif heuristic == "round_up_first":
        return [upper_bound, lower_bound]
    elif heuristic == "random":
        return [lower_bound, upper_bound] if np.random.rand() < 0.5 else [upper_bound, lower_bound]
    else:
        return [lower_bound, upper_bound]  # Default: round down first

# Branch & Bound algorithm with heuristics
def branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth, vbasis=[], cbasis=[], depth=0, var_selection="max_fractional", value_selection="round_down_first"):
    global nodes, lower_bound, upper_bound

    stack = deque()
    solutions = []
    solutions_found = 0
    best_sol_idx = 0

    best_sol_obj = -np.inf if isMax else np.inf

    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")
    nodes_per_depth[0] -= 1

    if DEBUG_MODE:
        debug_print()

    model.optimize()

    if model.status != GRB.OPTIMAL:
        if isMax:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], -np.inf, depth
        else:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], np.inf, depth

    x_candidate = model.getAttr('X', model.getVars())
    x_obj = model.ObjVal
    best_bound_per_depth[0] = x_obj

    selected_var_idx = select_branching_variable(x_candidate, integer_var, var_selection)
    
    if selected_var_idx is None:
        solutions.append([x_candidate, x_obj, depth])
        solutions_found += 1

        if DEBUG_MODE:
            debug_print(node=root_node, x_obj=x_obj, sol_status="Integer")
        return solutions, best_sol_idx, solutions_found

    branching_values = select_branching_order(x_candidate[selected_var_idx], value_selection)

    left_lb = np.copy(lb)
    left_ub = np.copy(ub)
    right_lb = np.copy(lb)
    right_ub = np.copy(ub)

    left_ub[selected_var_idx] = branching_values[0]
    right_lb[selected_var_idx] = branching_values[1]

    left_child = Node(left_ub, left_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
    right_child = Node(right_ub, right_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

    stack.append(right_child)
    stack.append(left_child)

    while stack:
        nodes += 1
        current_node = stack.pop()
        nodes_per_depth[current_node.depth] -= 1

        model.setAttr("LB", model.getVars(), current_node.lb)
        model.setAttr("UB", model.getVars(), current_node.ub)
        model.update()

        if DEBUG_MODE:
            debug_print()

        model.optimize()

        if model.status != GRB.OPTIMAL:
            if DEBUG_MODE:
                debug_print(node=current_node, sol_status="Infeasible")
            continue

        x_candidate = model.getAttr('X', model.getVars())
        x_obj = model.ObjVal

        selected_var_idx = select_branching_variable(x_candidate, integer_var, var_selection)

        if selected_var_idx is None:
            solutions.append([x_candidate, x_obj, current_node.depth])
            solutions_found += 1
            continue

        branching_values = select_branching_order(x_candidate[selected_var_idx], value_selection)

        left_lb = np.copy(current_node.lb)
        left_ub = np.copy(current_node.ub)
        right_lb = np.copy(current_node.lb)
        right_ub = np.copy(current_node.ub)

        left_ub[selected_var_idx] = branching_values[0]
        right_lb[selected_var_idx] = branching_values[1]

        left_child = Node(left_ub, left_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
        right_child = Node(right_ub, right_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

        stack.append(right_child)
        stack.append(left_child)

    if solutions_found > 0:
        best_sol_idx = min(range(len(solutions)), key=lambda i: solutions[i][1]) if not isMax else max(range(len(solutions)), key=lambda i: solutions[i][1])
        return solutions, best_sol_idx, solutions_found
    else:
        return [], -1, 0
