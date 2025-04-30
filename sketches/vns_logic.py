import numpy as np
import multiprocessing as mp # Use alias
from typing import List, Dict, Tuple, Union, Optional, Any, Iterable # Added Iterable
import itertools
import time # Optional: for demonstration/timing
import platform # To check OS
from functools import partial # To pass arguments with imap
import sys # Import sys for exiting if needed
import logging # Import the logging library

# ==============================================================================
# Helper Functions (Assume these are in 'functions.py' or defined here)
# ==============================================================================

# --- Option 1: Define directly (if not using functions.py) ---
# ... (Keep placeholder definitions or remove if using Option 2) ...

# --- Option 2: Import from functions.py (Recommended) ---
try:
    # Ensure functions.py is in the same directory or Python path
    # Make sure get_neighborhood_for_parallel is also in functions.py
    from functions import calculate_objective_serv_time_lookup, powerset, get_neighborhood_for_parallel
    # Use logging for success message after configuration
    # print("Successfully imported helper functions from functions.py")
except ImportError as e:
    # Use logging for error message after configuration
    # print(f"ERROR: Could not import required functions from functions.py: {e}")
    # print("Please ensure functions.py exists and contains the necessary function definitions.")
    # Option 1: Re-raise the error to stop execution
    # raise e # Raising error here prevents logging configuration below
    # Option 2: Log error and exit later or handle differently
    # Log the error *after* logging is configured in the main block
    pass # Handle import error after logging setup in main block

# ==============================================================================
# Worker Function (Takes all parameters directly)
# Must be defined at top-level or imported for multiprocessing.
# ==============================================================================

def _evaluate_neighbor(neighbor: np.ndarray, d: int, convolutions: Dict[int, np.ndarray], w: float) -> Tuple[Optional[float], np.ndarray]:
    """
    Worker function to evaluate a single neighbor solution.
    Accepts necessary parameters directly. Relies on calculate_objective_serv_time_lookup.
    Logs errors instead of printing.
    """
    try:
        # Calls the (potentially imported) objective function
        obj_results = calculate_objective_serv_time_lookup(neighbor, d, convolutions)
        waiting_time = float(obj_results[0])
        spillover = float(obj_results[1])
        cost = w * waiting_time + (1 - w) * spillover
        # Add NaN check for cost as well
        if cost != cost:
             # Use logging instead of print
             logging.error(f"Worker {mp.current_process().pid} calculated NaN cost for neighbor {neighbor}. Objectives: {obj_results}")
             return float('inf'), neighbor
        return cost, neighbor
    except Exception as e:
        # Use logging instead of print
        logging.error(f"Worker {mp.current_process().pid} failed evaluating neighbor {neighbor}. Error: {e}", exc_info=True) # Log traceback
        # Return a very high cost to prevent selection, and the neighbor for context
        return float('inf'), neighbor


# ==============================================================================
# Parallel Exploration Function (Uses worker function _evaluate_neighbor)
# Must be defined at top-level or imported.
# ==============================================================================

def explore_neighborhood_parallel(
    x_current: np.ndarray,
    c_current: float,
    t: int,
    d: int,
    convolutions: Dict[int, np.ndarray],
    w: float,
    v_star: np.ndarray,
    echo: bool = False # Echo still controls verbosity level
) -> Tuple[Optional[np.ndarray], Optional[float], bool]:
    """
    Explores the neighborhood of size 't' in parallel.
    Uses functools.partial to pass fixed arguments to the worker function.
    Relies on powerset and get_neighborhood_for_parallel.
    Attempts 'fork' on non-Windows, falls back to 'spawn'.
    Logs progress and errors instead of printing.
    """
    T = len(x_current)
    if t < 1 or t > T:
        # Use logging instead of print
        logging.warning(f"Neighborhood size t={t} is out of valid range [1, {T}]. Skipping.")
        return None, None, False

    if echo:
        # Use logging instead of print
        logging.info(f"  Exploring neighborhood t={t} in parallel (current cost: {c_current:.4f})...")

    # Generate indices iterator using the (potentially imported) powerset
    ids_gen = powerset(range(T), t) # Note: powerset generates indices from 0 to T-1

    # Generate the list of neighbor arrays using the (potentially imported) parallel function
    try:
        # Pass echo to the neighborhood generation function as well if needed for debugging
        # Assume get_neighborhood_for_parallel also uses logging if needed
        neighborhood = get_neighborhood_for_parallel(x_current, v_star, ids_gen, echo=echo)
        if not isinstance(neighborhood, list):
             raise TypeError(f"Neighborhood function did not return a list, got: {type(neighborhood)}")
    except Exception as e:
        # Use logging instead of print
        logging.error(f"Error during neighborhood generation for t={t}: {e}", exc_info=True)
        return None, None, False

    if len(neighborhood) == 0:
        if echo:
            # Use logging instead of print
            logging.info(f"  Neighborhood t={t} is empty. No exploration needed.")
        return None, None, False

    if echo:
        # Use logging instead of print
        neighbor_count = len(neighborhood)
        logging.info(f"  Size of neighborhood t={t}: {neighbor_count}")


    best_neighbor_found = None
    best_cost_found = c_current
    found_improvement = False

    # Limit number of workers if neighborhood is small to avoid overhead
    num_workers = min(max(1, mp.cpu_count() - 1), len(neighborhood))
    pool = None
    start_method = None # Determine start method below

    try:
        # --- Select Start Method ---
        if platform.system() != 'Windows':
            try:
                mp.get_context('fork')
                start_method = 'fork'
            except ValueError:
                start_method = 'spawn'
                if echo: logging.info(f"  'fork' not available, falling back to 'spawn' start method.")
        else:
            start_method = 'spawn' # 'spawn' is required on Windows

        # Log the selected start method if echo is True
        if echo: logging.info(f"  Using '{start_method}' start method for multiprocessing.")

        context = mp.get_context(start_method)
        pool = context.Pool(processes=num_workers) # No initializer

        # Create partial function to pass fixed args to the worker _evaluate_neighbor
        evaluate_func = partial(_evaluate_neighbor, d=d, convolutions=convolutions, w=w)

        # Map the partial function over the generated neighbors
        results_iterator = pool.imap_unordered(evaluate_func, neighborhood)

        processed_count = 0
        # --- Process results ---
        for cost, neighbor_result in results_iterator:
            processed_count += 1
            # Check for valid cost (not None, not NaN)
            if cost is None or cost != cost: # Checks for None and NaN
                if echo: logging.warning(f"  Received invalid cost ({cost}) for neighbor {neighbor_result}. Skipping.")
                continue

            # Check for improvement
            if cost < best_cost_found:
                best_cost_found = cost
                best_neighbor_found = neighbor_result # Capture the best neighbor array
                found_improvement = True
                if echo: logging.info(f"  Found potentially better solution (t={t}): cost {best_cost_found:.4f} (neighbor: {best_neighbor_found}). Stopping parallel search for t={t}.")
                # Terminate pool early on improvement
                if pool is not None:
                     pool.terminate() # Send signal to stop workers
                break # Exit results loop

        # --- Pool cleanup ---
        if pool is not None:
            if pool._state != mp.pool.TERMINATE: # Don't close if terminated
                pool.close()
            pool.join() # Wait for workers to exit

    except Exception as e:
        # Use logging instead of print
        logging.error(f"Error during parallel processing for t={t} (start_method='{start_method}'): {e}", exc_info=True)
        # Ensure pool is terminated and joined on error
        if pool is not None:
            try:
                pool.terminate()
                pool.join()
            except Exception as pool_e:
                 logging.error(f"Error during pool cleanup: {pool_e}", exc_info=True)
        return None, None, False # Indicate error / no improvement

    if found_improvement:
        # Ensure a valid neighbor array was captured
        if best_neighbor_found is None:
             # Use logging instead of print
             logging.warning(f"Improvement reported for t={t} but best_neighbor_found is None. Returning no improvement.")
             return None, None, False
        return best_neighbor_found, best_cost_found, True
    else:
        # Use logging instead of print
        if echo: logging.info(f"  No improvement found for t={t} after processing {processed_count} neighbors.")
        return None, None, False

# ==============================================================================
# Main VNS Function (Calls explore_neighborhood_parallel)
# Must be defined at top-level or imported.
# ==============================================================================

def variable_neighborhood_search(
    x_init: Union[List[int], np.ndarray],
    d: int,
    convolutions: Dict[int, np.ndarray],
    w: float,
    v_star: np.ndarray,
    max_t: Optional[int] = None,
    echo: bool = False # Echo still controls verbosity
) -> Tuple[np.ndarray, float]:
    """
    Performs Variable Neighborhood Search using parallel exploration for each 't'.
    Relies on calculate_objective_serv_time_lookup and explore_neighborhood_parallel.
    Logs progress and errors instead of printing.
    """
    try:
        x_star = np.array(x_init).flatten()
        if x_star.ndim != 1: raise ValueError("x_init must be 1D")
    except Exception as e:
        # Log error before raising
        logging.error(f"Invalid initial solution x_init: {e}", exc_info=True)
        raise ValueError(f"Invalid initial solution x_init: {e}") from e

    try:
        # Use the (potentially imported) objective function
        initial_objectives = calculate_objective_serv_time_lookup(x_star, d, convolutions)
        # Ensure initial objectives are valid floats
        obj1 = float(initial_objectives[0])
        obj2 = float(initial_objectives[1])
        if obj1 != obj1 or obj2 != obj2: # Check for NaN
             raise ValueError("Initial objective calculation resulted in NaN.")
        c_star = w * obj1 + (1 - w) * obj2
    except Exception as e:
         # Log error before raising
        logging.error(f"Failed to calculate valid initial cost for {x_star}: {e}", exc_info=True)
        raise ValueError(f"Failed to calculate valid initial cost for {x_star}: {e}") from e

    T = len(x_star)
    if T == 0:
        # Use logging instead of print
        logging.warning("Initial solution is empty.")
        return x_star, c_star # Return empty solution and its cost

    effective_max_t = max_t if max_t is not None and 0 < max_t <= T else T

    # Log initial state
    logging.info("-" * 50)
    logging.info("Starting Variable Neighborhood Search")
    logging.info(f"Initial solution: {x_star}, Initial cost: {c_star:.4f}")
    logging.info(f"Max neighborhood size (t): {effective_max_t}")
    logging.info(f"Number of parallel workers: {min(max(1, mp.cpu_count() - 1), T)}") # Revisit this calculation if needed
    logging.info("-" * 50)

    t = 1
    while t <= effective_max_t:
        # Use logging instead of print (controlled by echo)
        if echo: logging.info(f"\nAttempting neighborhood t={t} (current best cost: {c_star:.4f})")

        # Calls the parallel exploration function
        new_x, new_c, found_improvement_in_t = explore_neighborhood_parallel(
            x_current=x_star, c_current=c_star, t=t, d=d,
            convolutions=convolutions, w=w, v_star=v_star, echo=echo
        )

        # Check for valid improvement (found, cost is not None/NaN, cost is less than current)
        if found_improvement_in_t and new_c is not None and new_c == new_c and new_c < c_star:
            if echo:
                # Use logging instead of print
                logging.info(f"Improvement found! Updating best solution (t={t}).")
                logging.info(f"  Old cost: {c_star:.4f} -> New cost: {new_c:.4f}")
                logging.info(f"  New solution: {new_x}")
            # Ensure new_x is also valid before updating
            if new_x is not None and isinstance(new_x, np.ndarray):
                 x_star = new_x # Update best solution
                 c_star = new_c # Update best cost
                 t = 1 # Reset to first neighborhood
                 if echo: logging.info("Restarting search from t=1.")
            else:
                 # Use logging instead of print
                 logging.warning(f"Improvement reported for t={t} but new solution is invalid ({type(new_x)}). Continuing search without reset.")
                 t += 1 # Avoid infinite loop, move to next t
        else:
            # Handle cases where no improvement was found, or cost was invalid/not better
            if echo and not found_improvement_in_t:
                 # Use logging instead of print
                 logging.info(f"No improving solution found or error occurred for t={t}.")
            elif echo: # If found_improvement_in_t is True but condition failed
                 cost_repr = 'N/A' if new_c is None else ('NaN' if new_c != new_c else f"{new_c:.4f}")
                 # Use logging instead of print
                 logging.info(f"Neighbor found for t={t} but cost {cost_repr} is not better than current {c_star:.4f}.")
            t += 1 # Move to next neighborhood size

    # Log final results
    logging.info("-" * 50)
    logging.info("Variable Neighborhood Search finished.")
    logging.info(f"Final solution: {x_star}")
    logging.info(f"Final cost: {c_star:.4f}")
    logging.info("-" * 50)

    return x_star, c_star

# ==============================================================================
# Main Execution Block (MUST BE GUARDED FOR MULTIPROCESSING)
# ==============================================================================
if __name__ == "__main__":
    # --- Configure Logging ---
    log_file = 'vns_run.log'
    logging.basicConfig(
        level=logging.INFO,  # Log INFO level and above (INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
        filename=log_file,
        filemode='w'  # 'w' clears the file each time, 'a' appends
    )
    # Optional: Add a handler to also print logs to console
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setLevel(logging.INFO) # Or DEBUG
    # console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s')) # Simpler format for console
    # logging.getLogger().addHandler(console_handler)

    # --- Handle potential import error from earlier ---
    try:
        # Check if the required functions were actually imported or defined
        calculate_objective_serv_time_lookup
        powerset
        get_neighborhood_for_parallel
        logging.info("Successfully imported or defined helper functions.")
    except NameError:
        logging.error("Essential helper functions are missing. Exiting.")
        sys.exit("Exiting due to missing helper functions. Check imports or definitions.")


    # --- Crucial for 'spawn' start method (Windows, macOS, Quarto/Jupyter) ---
    mp.freeze_support() # Needed for frozen executables, good practice

    logging.info("Script execution started.")

    # --- Define Your Parameters ---
    N = 12 # Example: Number of patients
    T = 8  # Example: Number of intervals
    d = 5  # Example: Length of each interval
    w = 0.1 # Example: Weight for waiting time
    initial_solution = np.array([2, 1, 0, 1, 1, 0, 1, 6]) # Example initial schedule

    # --- Prepare Necessary Inputs ---
    # Replace these with your actual calculations/data loading
    logging.info("Preparing inputs (convolutions, v_star)...")
    # Example: Dummy convolutions (replace with your compute_convolutions)
    max_possible_patients = int(np.sum(initial_solution) + T) # Rough estimate
    dummy_convolutions = {i: np.random.rand(10)/10 for i in range(max_possible_patients + 1)}
    # Example: Dummy v_star (replace with your get_v_star)
    # Shape depends on how get_neighborhood_for_parallel uses it
    # Must have T columns if x has length T
    dummy_v_star = np.random.randint(-1, 2, size=(2*T, T)) # Example: 16 adjustment vectors for T=8

    logging.info("Inputs prepared. Starting VNS...")
    start_time = time.time()

    # --- Run the Variable Neighborhood Search ---
    # Ensure all functions called within VNS are defined above or imported
    best_solution, best_cost = variable_neighborhood_search(
        x_init=initial_solution,
        d=d,
        convolutions=dummy_convolutions, # Use your actual convolutions
        w=w,
        v_star=dummy_v_star,           # Use your actual v_star
        echo=True                      # Set echo=True for more detailed INFO logs
    )

    end_time = time.time()
    # Log final results summary
    logging.info("\n--- VNS Result ---")
    logging.info(f"Best solution found: {best_solution}")
    logging.info(f"Best cost found: {best_cost:.4f}")
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")

    logging.info("Script execution finished.")
    print(f"Script finished. Check log file: {log_file}") # Print final confirmation to console

