import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from collections import deque

# define global variables
isMax = None # max / min sense
DEBUG_MODE = True # debug enabled / disabled
nodes = 0 # number of nodes
lower_bound = -np.inf # lower bound of the problem
upper_bound = np.inf # upper bound of the problems

# calculate if a value is very close to an integer value
def is_nearly_integer(value, tolerance=1e-6):
    return abs(value - round(value)) <= tolerance

# Ορισμός της κλάσης Node
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



# Ευρετικές για cuts
def apply_cuts(model, cut_strategy):
    """
    Εφαρμόζει διαφορετικά cuts στο μοντέλο ανάλογα με τη στρατηγική.
    """
    if cut_strategy == "simple_cut":
        print("Εφαρμογή απλών cuts με όρια LB / UB.")
        # Δεν κάνουμε τίποτα, τα όρια διαχειρίζονται τα branches

    elif cut_strategy == "gomory_cut":
        print("Εφαρμογή Gomory Fractional Cuts μέσω Gurobi.")
        model.Params.Cuts = 2  # Ενεργοποίηση Gomory Cuts

    elif cut_strategy == "custom_cut":
        print("Εφαρμογή προσαρμοσμένων cuts.")
        for constr in model.getConstrs():
            model.addConstr(constr.Slack >= 1, name=f"custom_cut_{constr.ConstrName}")

    else:
        print(f"Μη έγκυρη στρατηγική cuts: {cut_strategy}. Χρήση default.")
    
    return model

# Branch & Bound με επιλογή cuts
def branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth, 
                     vbasis=[], cbasis=[], depth=0, var_selection="max_fractional", 
                     value_selection="round_down_first", cut_strategy="simple_cut"):
    """
    Branch & Bound με επιλογή heuristics και cuts.
    """
    global nodes, lower_bound, upper_bound

    stack = deque()
    solutions = []
    solutions_found = 0
    best_sol_idx = 0

    best_sol_obj = -np.inf if isMax else np.inf

    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")
    nodes_per_depth[0] -= 1

    print("Εφαρμογή στρατηγικής cuts:", cut_strategy)
    model = apply_cuts(model, cut_strategy)

    model.optimize()

    if model.status != GRB.OPTIMAL:
        if isMax:
            print("Το πρόβλημα είναι μη εφικτό.")
            return [], -np.inf, depth
        else:
            print("Το πρόβλημα είναι μη εφικτό.")
            return [], np.inf, depth

    x_candidate = model.getAttr('X', model.getVars())
    x_obj = model.ObjVal
    best_bound_per_depth[0] = x_obj

    selected_var_idx = None
    for idx, is_int_var in enumerate(integer_var):
        if is_int_var and not abs(x_candidate[idx] - round(x_candidate[idx])) <= 1e-6:
            selected_var_idx = idx
            break

    if selected_var_idx is None:
        solutions.append([x_candidate, x_obj, depth])
        solutions_found += 1
        return solutions, best_sol_idx, solutions_found

    branching_values = [np.floor(x_candidate[selected_var_idx]), np.ceil(x_candidate[selected_var_idx])]

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

        model.optimize()

        if model.status != GRB.OPTIMAL:
            continue

        x_candidate = model.getAttr('X', model.getVars())
        x_obj = model.ObjVal

        selected_var_idx = None
        for idx, is_int_var in enumerate(integer_var):
            if is_int_var and not abs(x_candidate[idx] - round(x_candidate[idx])) <= 1e-6:
                selected_var_idx = idx
                break

        if selected_var_idx is None:
            solutions.append([x_candidate, x_obj, current_node.depth])
            solutions_found += 1
            continue

        branching_values = [np.floor(x_candidate[selected_var_idx]), np.ceil(x_candidate[selected_var_idx])]

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
