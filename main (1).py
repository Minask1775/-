import time
import pandas as pd
from branch_and_bound import branch_and_bound
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def generate_supply_chain_problem(grid_size=10, total_points=20, num_clients=10, num_facilities=5):
    """
    Δημιουργεί ένα πρόβλημα αλυσίδας εφοδιασμού με τυχαία δεδομένα.
    """
    #np.random.seed(42)  # Για αναπαραγωγιμότητα

    # Τυχαία θέσεις πελατών και αποθηκών
    client_locations = np.random.rand(num_clients, 2) * grid_size
    facility_locations = np.random.rand(num_facilities, 2) * grid_size

    # Τυχαία σταθερά κόστη για κάθε αποθήκη
    fixed_costs = np.random.rand(num_facilities) * 100

    # Τυχαία κόστη μεταφοράς (απόσταση μεταξύ πελατών και αποθηκών)
    transport_costs = np.zeros((num_clients, num_facilities))
    for i in range(num_clients):
        for j in range(num_facilities):
            transport_costs[i][j] = np.linalg.norm(client_locations[i] - facility_locations[j])

    # Τυχαία ζήτηση πελατών
    demands = np.random.rand(num_clients) * 10

    return {
        "fixed_costs": fixed_costs,
        "transport_costs": transport_costs,
        "demands": demands,
        "num_clients": num_clients,
        "num_facilities": num_facilities,
    }

def create_pyomo_model(problem):
    """
    Δημιουργεί ένα μοντέλο Gurobi για το πρόβλημα αλυσίδας εφοδιασμού.
    """
    num_clients = problem["num_clients"]
    num_facilities = problem["num_facilities"]
    fixed_costs = problem["fixed_costs"]
    transport_costs = problem["transport_costs"]
    demands = problem["demands"]

    # Δημιουργία μοντέλου Gurobi
    model = gp.Model("SupplyChain")

    # Μεταβλητές απόφασης
    y = model.addVars(num_facilities, vtype=GRB.BINARY, name="y")  # y[j] = 1 αν η αποθήκη j είναι ενεργή
    x = model.addVars(num_clients, num_facilities, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")  # x[i,j] = ποσοστό ζήτησης

    # Στόχος: Ελαχιστοποίηση συνολικού κόστους
    model.setObjective(
        gp.quicksum(fixed_costs[j] * y[j] for j in range(num_facilities)) +
        gp.quicksum(transport_costs[i][j] * demands[i] * x[i, j] for i in range(num_clients) for j in range(num_facilities)),
        GRB.MINIMIZE
    )

    # Περιορισμοί
    # Κάθε πελάτης πρέπει να καλύπτεται πλήρως
    model.addConstrs(
        (gp.quicksum(x[i, j] for j in range(num_facilities)) == 1 for i in range(num_clients)),
        name="DemandCoverage"
    )

    # Μια αποθήκη μπορεί να καλύψει πελάτες μόνο αν είναι ενεργή
    model.addConstrs(
        (x[i, j] <= y[j] for i in range(num_clients) for j in range(num_facilities)),
        name="OpenWarehouse"
    )

    return model

def computational_study(num_classes=10, problems_per_class=10):
    """
    Εκτελεί μια υπολογιστική μελέτη για το πρόβλημα αλυσίδας εφοδιασμού.
    """
    results = []

    for cls in range(num_classes):
        for prob in range(problems_per_class):
            print(f"Δημιουργία προβλήματος {prob+1} στην κλάση {cls+1}...")
            problem = generate_supply_chain_problem(grid_size=10, total_points=20, num_clients=10, num_facilities=5)
            model = create_pyomo_model(problem)

            # Ορισμός των δεδομένων του προβλήματος για χρήση στον B&B
            fixed_costs = problem["fixed_costs"]
            transport_costs = problem["transport_costs"]
            demands = problem["demands"]
            num_vars = len(fixed_costs)  # Αριθμός μεταβλητών (αποθηκών)

            # Αρχικοποίηση των βοηθητικών δομών του B&B
            best_bound_per_depth = [float('inf')] * num_vars
            nodes_per_depth = [2**i for i in range(num_vars + 1)]

            # Αντί για Gurobi, καλούμε τον custom Branch and Bound
            start_time = time.time()
            solutions, best_sol_idx, solutions_found = branch_and_bound(
                model, fixed_costs, transport_costs, demands, best_bound_per_depth, nodes_per_depth
            )
            solve_time = time.time() - start_time

            # Αποθήκευση της βέλτιστης λύσης
            if solutions:
                optimal_solution = solutions[best_sol_idx][0]
                optimal_cost = solutions[best_sol_idx][1]
                print(f"Βέλτιστη λύση για πρόβλημα {prob+1} στην κλάση {cls+1}: {optimal_cost}")
                results.append({"class": cls, "problem": prob, "cost": optimal_cost, "time": solve_time})
            else:
                print(f"Πρόβλημα {prob+1} στην κλάση {cls+1} δεν επιλύθηκε βέλτιστα.")

    # Αποθήκευση αποτελεσμάτων σε CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv("results.csv", index=False)
        print("Τα αποτελέσματα αποθηκεύτηκαν στο 'results.csv'")
    else:
        print("Δεν αποθηκεύτηκαν αποτελέσματα. Δεν βρέθηκαν βέλτιστες λύσεις.")

if __name__ == "__main__":
    print("************************    Starting Computational Study...    ************************")

    # Καλούμε τη συνάρτηση computational_study
    computational_study(num_classes=10, problems_per_class=10)