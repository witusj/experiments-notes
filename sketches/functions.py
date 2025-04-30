import unittest
import numpy as np
from typing import List, Tuple, Dict, Iterable, TypeVar, Union
import copy
import random
from itertools import combinations

"""
These functions are used in the implementation of the optimization algorithm using local search for the scheduling problem.
For this two building blocks are necessary:
    - An evaluator: A set of functions whose task is to evaluate the objective function for a given solution (schedule).
    - A searcher: This set of functions that generate the neighborhood of a given solution and then explore the
      search space around the current solution using the evaluator.
"""

############################
# Helper functions
############################

def create_random_schedules(T, N, num_schedules):
  schedules = []
  for _ in range(num_schedules):
    sample = random.choices(range(T), k = N)
    schedule = np.bincount(sample, minlength=T).tolist()
    schedules.append(schedule)
  return(schedules)

def bailey_welch_schedule(T, d, N, s):
    """
    Generates a Bailey-Welch schedule for a clinic session.

    Parameters:
        T (int): Number of intervals in the clinic session.
        d (int): Length of each interval.
        N (int): Total number of patients to schedule.
        s (list of floats): Consultation time distribution where the index represents 
                            the consultation time (with 0 meaning no-show) and the value 
                            at each index is the probability of that consultation time.

    Returns:
        list: A list of length T where each element is the number of patients scheduled 
              in that interval.
    """
    # Compute the mean consultation time using the dot product.
    # Here, i represents the consultation time.
    mean_consultation = np.dot(np.arange(len(s)), s)
    
    # Initialize the schedule with T intervals.
    schedule = [0] * T

    # Schedule the first two patients at the beginning (interval 0)
    if N >= 1:
        schedule[0] += 1
    if N >= 2:
        schedule[0] += 1

    # For each subsequent patient, schedule them at intervals spaced by the mean consultation time.
    for i in range(2, N):
        # Calculate the ideal appointment time.
        appointment_time = (i - 1) * mean_consultation

        # Map the continuous appointment time to one of the T intervals.
        interval_index = int(appointment_time // d)

        # If the index exceeds available intervals, assign to the last interval.
        if interval_index >= T:
            interval_index = T - 1

        schedule[interval_index] += 1

    return schedule

def calculate_ambiguousness(y_pred_proba: np.ndarray) -> np.ndarray:
    """
    Calculate the ambiguousness (entropy) for each sample's predicted probabilities.

    The ambiguousness is calculated as the entropy of the predicted class probabilities.

    Parameters:
    y_pred_proba (np.ndarray): Array of shape (n_samples, n_classes) with predicted probabilities for each class.

    Returns:
    np.ndarray: Array of ambiguousness (entropy) for each sample.
    """
    # Ensure probabilities are in numpy array
    y_pred_proba = np.array(y_pred_proba)

    # Define a small bias term to avoid log(0)
    epsilon = 1e-10

    # Add small bias term to probabilities that contain zeros
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

    # Calculate ambiguousness (entropy) for each sample
    ambiguousness = -np.sum(y_pred_proba * np.log2(y_pred_proba), axis=1)

    return ambiguousness
  
def compare_json(json1: dict, json2: dict) -> dict:
    """
    Compare two JSON objects and return a dictionary with the differences.

    Parameters:
    json1 (dict): The first JSON object to compare.
    json2 (dict): The second JSON object to compare.

    Returns:
    dict: A dictionary showing the differences between the two JSON objects.
    """
    differences = {}

    # Check keys in json1
    for key in json1.keys():
        if key in json2:
            if json1[key] != json2[key]:
                differences[key] = {
                    "json1_value": json1[key],
                    "json2_value": json2[key]
                }
        else:
            differences[key] = {
                "json1_value": json1[key],
                "json2_value": "Key not found in json2"
            }

    # Check keys in json2 that are not in json1
    for key in json2.keys():
        if key not in json1:
            differences[key] = {
                "json1_value": "Key not found in json1",
                "json2_value": json2[key]
            }

    return differences

############################
# Evaluator functions
############################

def service_time_with_no_shows(s: List[float], q: float) -> List[float]:
    """
    Adjust a distribution of service times for no-shows.

    The adjusted service time distribution accounts for the probability q of no-shows.

    Parameters:
        s (List[float]): The original service time probability distribution.
        q (float): The no-show probability.

    Returns:
        List[float]: The adjusted service time distribution.
    """
    # Adjust the service times by multiplying with (1 - q)
    s_adj = [(1 - q) * si for si in s]
    # Add the no-show probability to the zero service time
    s_adj[0] += q
    return s_adj


def compute_convolutions(probabilities: List[float], N: int, q: float = 0.0) -> Dict[int, np.ndarray]:
    """
    Computes the k-fold convolution of a probability mass function (PMF) for k in range 1 to N.
    
    Before performing the convolutions, the PMF is adjusted for no-shows using the `service_time_with_no_shows` function.
    Convolution is performed using NumPy's `np.convolve`.
    
    Parameters:
        probabilities (List[float]): The original PMF represented as a list where the index is the value (e.g., service time) 
                                      and the value is the corresponding probability.
        N (int): The maximum number of convolutions to compute.
        q (float, optional): The no-show probability. Defaults to 0.0.
        
    Returns:
        Dict[int, np.ndarray]: A dictionary where each key k (1 ≤ k ≤ N) corresponds to the PMF resulting from the k-fold convolution
                                 of the adjusted PMF.
    """
    probs_adj_q = service_time_with_no_shows(probabilities, q)
    convolutions: Dict[int, np.ndarray] = {1: np.array(probs_adj_q, dtype=np.float64)}
    for k in range(2, N + 1):
        convolutions[k] = np.convolve(convolutions[k - 1], probs_adj_q)
    return convolutions


def fft_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Convolve two 1-D arrays using FFT.
    
    Parameters:
        a (np.ndarray): First input array.
        b (np.ndarray): Second input array.
        
    Returns:
        np.ndarray: The convolution result.
    """
    n = len(a) + len(b) - 1
    # Use the next power of 2 for efficiency
    n_fft = 2**int(np.ceil(np.log2(n)))
    A = np.fft.rfft(a, n=n_fft)
    B = np.fft.rfft(b, n=n_fft)
    conv_result = np.fft.irfft(A * B, n=n_fft)[:n]
    return conv_result


def compute_convolutions_fft(probabilities: Union[List[float], np.ndarray], N: int, q: float = 0.0) -> Dict[int, np.ndarray]:
    """
    Computes the k-fold convolution of a PMF using FFT-based convolution.
    
    Parameters:
        probabilities (Union[List[float], np.ndarray]): The PMF.
        N (int): Maximum number of convolutions.
        q (float): Parameter for adjusting the PMF.
        
    Returns:
        Dict[int, np.ndarray]: Keys are k (number of convolutions), values are the convolved PMFs.
    """
    probs_adj_q = service_time_with_no_shows(list(probabilities), q)
    convolutions: Dict[int, np.ndarray] = {1: np.array(probs_adj_q, dtype=np.float64)}
    for k in range(2, N + 1):
        convolutions[k] = fft_convolve(convolutions[k - 1], probs_adj_q)
    return convolutions


def calculate_objective_serv_time_lookup(schedule: List[int], d: int, convolutions: Dict[int, np.ndarray]) -> Tuple[float, float]:
    """
    Calculate the objective value based on the given schedule and parameters using precomputed convolutions.
    
    This function calculates two key performance metrics:
    
    - **ewt**: The sum of expected waiting times across time slots.
    - **esp**: The expected spillover time (or overtime) after the final time slot.
    
    Parameters:
        schedule (List[int]): A list representing the number of patients scheduled in each time slot.
        d (int): Duration threshold for a time slot.
        convolutions (Dict[int, np.ndarray]): A dictionary of precomputed convolutions of the service time PMF.
                                                The key `1` contains the adjusted service time distribution.
    
    Returns:
        Tuple[float, float]: A tuple containing:
            - ewt (float): The sum of expected waiting times across all time slots.
            - esp (float): The expected spillover time (overtime) at the end of the schedule.
    """
    # Create initial service process with probability 1 at time 0
    sp = np.zeros(d + 1, dtype=np.float64)
    sp[0] = 1.0  # The probability that the service time is zero is 1.
    
    ewt = 0.0  # Total expected waiting time
    conv_s = convolutions.get(1)
    est = np.dot(range(len(conv_s)), conv_s)  # Expected service time
    
    for x in schedule:
        if x == 0:
            # No patients in this time slot: advance time by d
            if len(sp) > d:
                sp[d] = np.sum(sp[:d + 1])
                sp = sp[d:]
            else:
                sp = np.array([np.sum(sp)], dtype=np.float64)
        else:
            # Patients are scheduled in this time slot
            esp = np.dot(range(len(sp)), sp)  # Expected spillover before this time slot
            
            # Calculate waiting time:
            ewt += x * esp + est * (x - 1) * x / 2
            
            # Update the waiting time distribution after serving all patients in this slot
            sp = np.convolve(sp, convolutions.get(x))
            
            # Adjust for the duration threshold d
            if len(sp) > d:
                sp[d] = np.sum(sp[:d + 1])
                sp = sp[d:]
            else:
                sp = np.array([np.sum(sp)], dtype=np.float64)
    
    # Expected spillover time at the end
    esp = np.dot(range(len(sp)), sp)
    
    return ewt, esp

############################
# Searcher functions
############################

T = TypeVar('T')  # Generic type variable

def powerset(iterable: Iterable[int], size: int = 1) -> List[List[int]]:
    """
    Generate all subsets of a given size from the input iterable.

    Parameters:
        iterable (Iterable[int]): The input iterable containing integers.
        size (int): The size of each subset to generate (default is 1).

    Returns:
        List[List[int]]: A list where each element is a list representing a subset of the given size.

    Example:
        powerset([1, 2, 3], 2) returns [[1, 2], [1, 3], [2, 3]]
    """
    # Use itertools.combinations to generate subsets of the specified size,
    # then convert each combination (tuple) to a list.
    return [[i for i in item] for item in combinations(iterable, size)]


def get_v_star(T: int) -> np.ndarray:
    """
    Generate a set of vectors V* of length T, where each vector is a cyclic permutation of an initial vector.

    The initial vector 'u' is defined as:
      - u[0] = -1
      - u[-1] = 1
      - all other elements are 0

    Parameters:
        T (int): Length of the vectors.

    Returns:
        np.ndarray: An array of shape (T, T), where each row is a vector in V*.
    """
    # Create an initial vector 'u' of zeros with length T
    u = np.zeros(T, dtype=np.int64)
    u[0] = -1
    u[-1] = 1
    v_star = [u.copy()]

    # Generate cyclic permutations of 'u'
    for i in range(T - 1):
        u = np.roll(u, 1)
        v_star.append(u.copy())

    return np.array(v_star)


def get_neighborhood(x: Union[List[int], np.ndarray],
                     v_star: np.ndarray,
                     ids: List[List[int]],
                     echo: bool = False) -> np.ndarray:
    """
    Generate the neighborhood of a solution by adding combinations of vectors from v_star to x.

    Parameters:
        x (Union[List[int], np.ndarray]): The current solution vector.
        v_star (np.ndarray): An array where each row is a vector used for adjustments.
        ids (List[List[int]]): A list of index lists, each specifying which rows in v_star to combine.
        verbose (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        np.ndarray: An array of neighbor solutions (rows) with non-negative entries.
    """
    x = np.array(x)  # Convert input solution to a NumPy array
    p = 50  # Print every pth result if verbose
    if echo:
        print(f"Printing every {p}th result")
    neighborhood = []  # To store the generated neighbor solutions

    # Loop over each combination of indices in ids
    for i in range(len(ids)):
        neighbor = np.zeros(len(x), dtype=int)
        for j in range(len(ids[i])):
            if echo:
                print(f"v_star[{ids[i][j]}]: {v_star[ids[i][j]]}")
            neighbor += v_star[ids[i][j]]
        x_n = x + neighbor
        if i % p == 0 and echo:
            print(f"x, x', delta:\n{x},\n{x_n},\n{neighbor}\n-----------------")
        neighborhood.append(x_n)
    
    neighborhood = np.array(neighborhood)
    if echo:
        print(f"Size of raw neighborhood: {len(neighborhood)}")
    # Filter out neighbors that contain negative values
    mask = ~np.any(neighborhood < 0, axis=1)
    if echo:
        print(f"Filtered out: {len(neighborhood) - mask.sum()} schedules with negative values.")
    filtered_neighborhood = neighborhood[mask]
    if echo:
        print(f"Size of filtered neighborhood: {len(filtered_neighborhood)}")
    return filtered_neighborhood
  
def get_neighborhood_for_parallel(
    x: Union[List[int], np.ndarray],
    v_star: np.ndarray,
    ids_gen: Iterable[Tuple[int, ...]], # Accepts iterator of index tuples
    echo: bool = False
) -> List[np.ndarray]: # Returns list of 1D neighbor arrays
    """
    Generates the neighborhood for parallel processing by adding combinations
    of vectors from v_star to x.

    This version is adapted for use with the parallel VNS, accepting an
    iterator of index tuples and returning a list of valid neighbor arrays.

    Parameters:
        x (Union[List[int], np.ndarray]): The current solution vector (1D).
        v_star (np.ndarray): An array where each row is a vector used for adjustments.
                             Shape should be (num_adjustments, len(x)).
        ids_gen (Iterable[Tuple[int, ...]]): An iterator yielding tuples of indices,
                                             each specifying which rows in v_star to combine.
        echo (bool, optional): If True, prints debug information. Defaults to False.

    Returns:
        List[np.ndarray]: A list where each element is a valid (non-negative)
                          neighbor solution vector (1D np.ndarray). Returns an
                          empty list if no valid neighbors are generated.
    """
    try:
        x_base = np.array(x).flatten()
        if x_base.ndim != 1:
             raise ValueError("Input solution 'x' must be a 1D array or convertible to one.")
        T = len(x_base) # Length of the solution vector
    except Exception as e:
        print(f"Error processing input solution x: {e}")
        return [] # Return empty list if input is invalid

    if v_star.ndim != 2 or (v_star.size > 0 and v_star.shape[1] != T):
         print(f"Warning: v_star shape {v_star.shape} is incompatible with x length {T}. Check v_star generation.")
         # Depending on logic, might return [] or proceed assuming zero adjustments
         # For safety, returning empty list if dimensions mismatch significantly
         if v_star.size > 0 and v_star.shape[1] != T:
              return []

    neighbors_list = []  # Initialize an empty list to store valid neighbors
    p = 50  # Print every pth result if verbose (can adjust or remove)
    count = 0

    if echo:
        print(f"Generating neighbors for parallel processing (printing every {p}th attempt if echo=True)")

    # Loop over each tuple of indices yielded by the iterator ids_gen
    for indices_tuple in ids_gen:
        count += 1
        if not indices_tuple: # Skip if the tuple is empty (e.g., from powerset(..., 0))
            continue

        # Calculate the combined adjustment vector for this tuple
        try:
            # Ensure indices are valid for v_star dimensions
            valid_indices = [idx for idx in indices_tuple if 0 <= idx < v_star.shape[0]]
            if not valid_indices: # If no valid indices in tuple, skip or use zero adjustment
                 adjustment_vector = np.zeros(T, dtype=int)
                 if echo and indices_tuple: print(f"Warning: Indices {indices_tuple} out of bounds for v_star (shape {v_star.shape}). Using zero adjustment.")
            elif v_star.size == 0: # Handle empty v_star explicitly
                 adjustment_vector = np.zeros(T, dtype=int)
            else:
                 # Sum the valid rows from v_star
                 adjustment_vector = np.sum(v_star[valid_indices, :], axis=0, dtype=int)

        except IndexError as e:
             # This might happen if v_star is unexpectedly empty or dimensions changed
             if echo:
                 print(f"Warning: IndexError calculating adjustment for indices {indices_tuple}. v_star shape: {v_star.shape}. Error: {e}. Using zero adjustment.")
             adjustment_vector = np.zeros(T, dtype=int)
             continue # Skip this problematic combination
        except Exception as e:
             # Catch other potential errors during adjustment calculation
             if echo:
                 print(f"Error calculating adjustment for indices {indices_tuple}: {e}. Using zero adjustment.")
             adjustment_vector = np.zeros(T, dtype=int)
             continue # Skip this problematic combination


        # Apply the adjustment to the base solution
        neighbor_candidate = x_base + adjustment_vector

        if count % p == 0 and echo:
            print(f"Attempt {count}: Base x: {x_base}, Adjustment: {adjustment_vector} (Indices: {indices_tuple}), Candidate: {neighbor_candidate}")

        # Check if the resulting neighbor is valid (non-negative)
        # Add any other domain-specific validity checks here if needed
        if np.all(neighbor_candidate >= 0):
            neighbors_list.append(neighbor_candidate) # Add the valid 1D array to the list
            if count % p == 0 and echo:
                 print(" -> Valid neighbor added.")
        elif echo and count % p == 0:
             print(f" -> Invalid neighbor (negative values: {neighbor_candidate[neighbor_candidate < 0]}). Discarded.")


    if echo:
        print(f"Finished generating neighbors. Total valid neighbors found: {len(neighbors_list)}")

    return neighbors_list # Return the list of valid neighbor arrays

def build_welch_bailey_schedule(N, T):
    """
    Build a schedule based on the Welch and Bailey (1952) heuristic.

    Parameters:
    N (int): Number of patients to be scheduled.
    T (int): Number of time intervals in the schedule.

    Returns:
    list: A schedule of length T where each item represents the number of patients scheduled
          at the corresponding time interval.
    """
    # Initialize the schedule with zeros
    schedule = [0] * T

    # Schedule the first two patients at the beginning
    schedule[0] = 2
    remaining_patients = N - 2

    # Distribute patients in the middle time slots with gaps
    for t in range(1, T - 1):
        if remaining_patients <= 0:
            break
        if t % 2 == 1:  # Create gaps (only schedule patients at odd time slots)
            schedule[t] = 1
            remaining_patients -= 1

    # Push any remaining patients to the last time slot
    schedule[-1] += remaining_patients

    return schedule

def local_search(x: Union[List[int], np.ndarray],
                 d: int,
                 convolutions: Dict[int, np.ndarray],
                 w: float,
                 v_star: np.ndarray,
                 size: int = 2,
                 echo: bool = False) -> Tuple[np.ndarray, float]:
    """
    Perform local search to optimize a schedule.

    Parameters:
        x (Union[List[int], np.ndarray]): The initial solution vector.
        d (int): Duration threshold for a time slot.
        convolutions (Dict[int, np.ndarray]): Precomputed convolutions of the service time PMF.
        w (float): Weighting factor for combining the objectives.
        v_star (np.ndarray): Array of adjustment vectors.
        size (int, optional): Maximum number of patients to switch in a neighborhood (default is 2).
        echo (bool, optional): If True, prints progress messages. Defaults to False.

    Returns:
        Tuple[np.ndarray, float]: A tuple containing the best solution found and its associated cost.
    """
    x_star = np.array(x).flatten()  # Best solution (1D array)
    objectives_star = calculate_objective_serv_time_lookup(x_star, d, convolutions)
    c_star = w * objectives_star[0] + (1 - w) * objectives_star[1]
    print(f"Initial solution: {x_star}, cost: {c_star}")
    T = len(x_star)  # Length of the solution vector
    N = np.sum(x_star)  # Total number of patients

    t = 1
    while t < size:
        if echo:
            print(f'Running local search with switching {t} patient(s)')

        # Generate neighborhood by switching t patients
        ids_gen = powerset(range(T), t)
        neighborhood = get_neighborhood(x_star, v_star, ids_gen)
        if echo:
            print(f"Size of neighborhood: {len(neighborhood)}")

        found_better_solution = False

        for neighbor in neighborhood:
            waiting_time, spillover = calculate_objective_serv_time_lookup(neighbor, d, convolutions)
            cost = w * waiting_time + (1 - w) * spillover
            if cost < c_star:
                x_star = neighbor
                c_star = cost
                if echo:
                    print(f"Found better solution: {x_star}, cost: {c_star}")
                found_better_solution = True
                break

        if found_better_solution:
            t = 1  # Restart search with t = 1 if improvement is found
        else:
            t += 1  # Increase neighborhood size if no improvement

    return x_star, c_star
