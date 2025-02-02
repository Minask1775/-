import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from collections import deque
import problems as pr

# define global variables
isMax = None # max / min sense
DEBUG_MODE = True # debug enabled / disabled
nodes = 0 # number of nodes
lower_bound = -np.inf # lower bound of the problem
upper_bound = np.inf # upper bound of the problems

# calculate if a value is very close to an integer value
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


# branch & bound algorithm
def branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth, vbasis=[], cbasis=[], depth=0):
    global nodes, lower_bound, upper_bound

    # create stack using deque() structure
    stack = deque()

    # initialize solution list
    solutions = list()
    solutions_found = 0
    best_sol_idx = 0

    # initialize best solution
    if isMax:
        best_sol_obj = -np.inf
    else:
        best_sol_obj = np.inf

    # create root node
    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")
    nodes_per_depth[0] -= 1

    # ===============  Root node  ==========================
    if DEBUG_MODE:
        debug_print()

    # solve relaxed problem
    model.optimize()

    # check if the model was solved to optimality. If not then return (infeasible).
    if model.status != GRB.OPTIMAL:
        if isMax:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], -np.inf, depth
        else:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], np.inf, depth

    # get the solution (variable assignments)
    x_candidate = model.getAttr('X', model.getVars())

    # get the objective value
    x_obj = model.ObjVal
    best_bound_per_depth[0] = x_obj

    # check if all variables have integer values (from the ones that are supposed to be integers).
    # If not, then select the first variable with a fractional value to be the one fixed
    vars_have_integer_vals = True
    for idx, is_int_var in enumerate(integer_var):
        if is_int_var and not is_nearly_integer(x_candidate[idx]):
            vars_have_integer_vals = False
            selected_var_idx = idx
            break

    # found feasible solution
    if vars_have_integer_vals:
        # if we have a feasible solution in root, then terminate
        solutions.append([x_candidate, x_obj, depth])
        solutions_found += 1

        if DEBUG_MODE:
            debug_print(node=root_node, x_obj=x_obj, sol_status="Integer")
        return solutions, best_sol_idx, solutions_found

    # otherwise update lower/upper bound for min/max respectively
    else:
        if isMax:
            upper_bound = x_obj
        else:
            lower_bound = x_obj

    if DEBUG_MODE:
        debug_print(node=root_node, x_obj=x_obj, sol_status="Fractional")

    # Retrieve vbasis and cbasis
    vbasis = model.getAttr("VBasis", model.getVars())
    cbasis = model.getAttr("CBasis", model.getConstrs())

    # create lower bounds and upper bounds for the variables of the child nodes
    left_lb = np.copy(lb)
    left_ub = np.copy(ub)
    right_lb = np.copy(lb)
    right_ub = np.copy(ub)

    # create left and right branches (e.g. set left: x = 0, right: x = 1 in a binary problem)
    left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
    right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

    # create child nodes
    left_child = Node(left_ub, left_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
    right_child = Node(right_ub, right_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

    # add child nodes in stack
    stack.append(right_child)
    stack.append(left_child)

    # solving sub problems
    # While the stack has nodes, continue solving
    while (len(stack) != 0):
        print("\n********************************  NEW NODE BEING EXPLORED  ******************************** ")

        # increment total nodes by 1
        nodes += 1

        # get the child node on top of stack
        current_node = stack[-1]

        # remove this node from stack
        stack.pop()

        # increase the nodes visited for current depth
        nodes_per_depth[current_node.depth] -= 1

        # warm start solver. Use the vbasis and cbasis that parent node passed to the current one.
        if (len(current_node.vbasis) != 0) and (len(current_node.cbasis) != 0):
            model.setAttr("VBasis", model.getVars(), current_node.vbasis)
            model.setAttr("CBasis", model.getConstrs(), current_node.cbasis)

        # update the state of the model, passing the new lower bounds/upper bounds for the vars.
        # Basically, we only change the ub/lb for the branching variable. Another way is to introduce a new constraint (e.g. x_i <= ub).
        model.setAttr("LB", model.getVars(), current_node.lb)
        model.setAttr("UB", model.getVars(), current_node.ub)
        model.update()

        if DEBUG_MODE:
            debug_print()

        # optimize the model
        model.optimize()

        # Check if the model was solved to optimality. If not then do not create child nodes.
        infeasible = False
        if model.status != GRB.OPTIMAL:
            if isMax:
                infeasible = True
                x_obj = -np.inf
            else:
                infeasible = True
                x_obj = np.inf
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 * (i - current_node.depth)

        else:
            # get the solution (variable assignments)
            x_candidate = model.getAttr('X', model.getVars())

            # get the objective value
            x_obj = model.ObjVal

            # update best bound per depth if a better solution was found
            if isMax == True and x_obj > best_bound_per_depth[current_node.depth]:
                best_bound_per_depth[current_node.depth] = x_obj
            elif isMax == False and x_obj < best_bound_per_depth[current_node.depth]:
                best_bound_per_depth[current_node.depth] = x_obj

            # if we reached the final node of a depth, then update the bounds
            if nodes_per_depth[current_node.depth] == 0:
                if isMax == True:
                    upper_bound = best_bound_per_depth[current_node.depth]
                else:
                    lower_bound = best_bound_per_depth[current_node.depth]

        # if infeasible don't create children (continue searching the next node)
        if infeasible:
            if DEBUG_MODE:
                debug_print(node=current_node, sol_status="Infeasible")
            continue

        # check if all variables have integer values (from the ones that are supposed to be integers)
        vars_have_integer_vals = True
        for idx, is_int_var in enumerate(integer_var):
            if is_int_var and not is_nearly_integer(x_candidate[idx]):
                vars_have_integer_vals = False
                selected_var_idx = idx
                break

        # found feasible solution
        if vars_have_integer_vals: # integer solution
            if isMax:
                if lower_bound < x_obj: # a better solution was found
                    lower_bound = x_obj
                    if abs(lower_bound - upper_bound) < 1e-6: # optimal solution
                        # store solution, number of solutions and best sol index (and return)
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if (abs(x_obj - best_sol_obj) < 1e-6) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1

                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        return solutions, best_sol_idx, solutions_found

                    # Not optimal. Store solution, number of solutions and best sol index (and do not expand children)
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    # remove the children nodes from each next depth
                    for i in range(current_node.depth + 1, len(nodes_per_depth)):
                        nodes_per_depth[i] -= 2 * (i - current_node.depth)

                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                    continue

            else:
                if upper_bound > x_obj: # better solution
                    upper_bound = x_obj # update bound
                    if abs(lower_bound - upper_bound) < 1e-6: # optimality reached
                        # store solution, number of solutions and best sol index (and return)
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1
                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        return solutions, best_sol_idx, solutions_found

                    # Not optimal. Store solution, number of solutions and best sol index (and do not expand children)
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    # remove the children nodes from each next depth
                    for i in range(current_node.depth + 1, len(nodes_per_depth)):
                        nodes_per_depth[i] -= 2 * (i - current_node.depth)

                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                    continue

            # do not branch further if is an equal solution
            # remove the children nodes from each next depth
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 * (i - current_node.depth)

            if DEBUG_MODE:
                debug_print(node=current_node, x_obj=x_obj,
                            sol_status="Integer (Rejected -- Doesn't improve incumbent)")
            continue

        if isMax:
            if x_obj < lower_bound or abs(x_obj - lower_bound) < 1e-6: # cut
                # remove the children nodes from each next depth
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 * (i - current_node.depth)
                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                continue
        else:
            if x_obj > upper_bound or abs(x_obj - upper_bound) < 1e-6: # cut
                # remove the children nodes from each next depth
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 * (i - current_node.depth)
                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                continue

        if DEBUG_MODE:
            debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional")

        # Retrieve vbasis and cbasis
        vbasis = model.getAttr("VBasis", model.getVars())
        cbasis = model.getAttr("CBasis", model.getConstrs())

        # create lower bounds and upper bounds for child nodes
        left_lb = np.copy(current_node.lb)
        left_ub = np.copy(current_node.ub)
        right_lb = np.copy(current_node.lb)
        right_ub = np.copy(current_node.ub)

        # create left and right branches  (e.g. set left: x = 0, right: x = 1 in a binary problem)
        left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
        right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

        # create child nodes
        left_child = Node(left_ub, left_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx,
                          "Left")
        right_child = Node(right_ub, right_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx,
                           "Right")

        # add child nodes in stack
        stack.append(right_child)
        stack.append(left_child)

    return solutions, best_sol_idx, solutions_found


if __name__ == "__main__":

    print("************************    Initializing structures...    ************************")

    model, ub, lb, integer_var, num_vars, c = pr.pcenter("30_5_10/0.txt")
    isMax = False

    # Initialize structures
    # Keep the best bound per depth and the total nodels visited for each depth
    if isMax == True:
        best_bound_per_depth = np.array([-np.inf for i in range(num_vars)])
    else:
        best_bound_per_depth = np.array([np.inf for i in range(num_vars)])

    nodes_per_depth = np.zeros(num_vars + 1, dtype=float)
    nodes_per_depth[0] = 1
    for i in range(1, num_vars + 1):
        nodes_per_depth[i] = nodes_per_depth[i - 1] * 2

    # Start solving
    print("************************    Solving problem...    ************************")
    start = time.time()
    solutions, best_sol_idx, solutions_found = branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth,
                                                                nodes_per_depth)
    end = time.time()

    # Print results
    print("========= Optimal Solutions =========")
    """print("Variable:")
    for i in range(len(solutions[-2][0])): 
        print(f">  x{i} = {solutions[-2][0][i]}")

    print(f"Objective Value: {solutions[-2][1]}")
    print(f"Tree depth: {solutions[-2][2]}")"""

    print(solutions[best_sol_idx][0])
    print(f"Objective Value: {solutions[best_sol_idx][1]}")
    print(f"Tree depth: {solutions[best_sol_idx][2]}")
    print()
    print(solutions)
    print(f"Time Elapsed: {end - start}")
    print(f"Total nodes: {nodes}")

