import unittest
import numpy as np
from typing import List, Tuple, Dict
import copy

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

def compute_convolutions(probabilities, N, q=0.0):
    """
    Computes the k-fold convolution of a probability mass function (PMF) for k in range 1 to N.
    
    Before performing the convolutions, the PMF is adjusted for no-shows using the `service_time_with_no_shows` function.
    Convolution is performed using NumPy's `np.convolve`.
    
    Parameters:
        probabilities (List[float]): The original PMF represented as a list where the index is the value (e.g., service time) 
                                      and the value is the corresponding probability. This function is generic and can be used
                                      for any PMF, not just for service times.
        N (int): The maximum number of convolutions to compute.
        q (float, optional): The no-show probability. Defaults to 0.0.
        
    Returns:
        Dict[int, np.ndarray]: A dictionary where each key k (1 ≤ k ≤ N) corresponds to the PMF resulting from the k-fold convolution
                                 of the adjusted PMF.
    """
    convolutions = {}
    probs_adj_q = service_time_with_no_shows(probabilities, q)
    convolutions = {1: probs_adj_q}
    for k in range(2, N + 1):
        convolutions[k] = np.convolve(convolutions[k - 1], probs_adj_q)
    return convolutions

def fft_convolve(a, b):
    """
    Convolve two 1-D arrays using FFT.
    
    Parameters:
        a, b (np.array): Input arrays.
        
    Returns:
        np.array: The convolution result.
    """
    n = len(a) + len(b) - 1
    # Use the next power of 2 for efficiency
    n_fft = 2**int(np.ceil(np.log2(n)))
    A = np.fft.rfft(a, n=n_fft)
    B = np.fft.rfft(b, n=n_fft)
    conv_result = np.fft.irfft(A * B, n=n_fft)[:n]
    return conv_result

def compute_convolutions_fft(probabilities, N, q=0.0):
    """
    Computes the k-fold convolution of a PMF using FFT-based convolution.
    
    Parameters:
        probabilities (list or np.array): The PMF.
        N (int): Maximum number of convolutions.
        q (float): Parameter for adjusting the PMF.
        
    Returns:
        dict: Keys are k (number of convolutions), values are the convolved PMFs.
    """
    convolutions = {}
    probs_adj_q = service_time_with_no_shows(probabilities, q)
    convolutions[1] = probs_adj_q
    for k in range(2, N + 1):
        convolutions[k] = fft_convolve(convolutions[k - 1], probs_adj_q)
    return convolutions

def calculate_objective_serv_time_lookup(schedule: List[int], d: int, convolutions: dict) -> Tuple[float, float]:
    """
    Calculate the objective value based on the given schedule and parameters using precomputed convolutions.
    
    This function calculates two key performance metrics:
    
    - **ewt**: The sum of expected waiting times across time slots.
    - **esp**: The expected spillover time (or overtime) after the final time slot.
    
    The function assumes that the convolutions dictionary contains precomputed convolutions of the service time 
    probability mass function (PMF). The key `1` corresponds to the adjusted service time distribution (for instance,
    one adjusted for no-shows), and keys corresponding to the number of patients in a time slot (e.g., `x`) are used 
    for computing the waiting time distribution when `x` patients are scheduled.
    
    The calculation proceeds by iteratively updating a spillover time distribution, `sp`, which is initially 
     zero with probability 1. For each time slot:
      - If no patients are scheduled, the service process is advanced by `d`.
      - If patients are scheduled, the expected waiting time is updated based on the current spillover
        and the expected service time. Then the waiting time distribution is updated via convolution with the PMF for the
        given number of patients, and adjusted for the duration threshold `d`.
    
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
    sp[0] = 1.0  # Initial service process (no waiting time) The probability that the service time is zero is 1.
    
    ewt = 0  # The initial total expected waiting time
    conv_s = convolutions.get(1)  # Adjusted service time distribution
    est = np.dot(range(len(conv_s)), conv_s)  # Expected service time
    
    for x in schedule:
        if x == 0:
            # No patients in this time slot
            # Handle the case when there are zero patients scheduled by advancing the time by d
            if len(sp) > d:
                sp[d] = np.sum(sp[:d + 1])
                sp = sp[d:]
            else:
                # If sp is shorter than d, create a new array with one element (sum of all probabilities)
                sp = np.array([np.sum(sp)], dtype=np.float64)
        else:
            # Patients are scheduled in this time slot
            esp = np.dot(range(len(sp)), sp)  # Expected spillover before this time slot
            
            # Calculate waiting time:
            # - First patient: expected spillover from previous slot
            # - For x patients: first patient's wait + sum of (x-1) additional patient wait times
            ewt += x * esp + est * (x - 1) * x / 2
            
            # Update the waiting time distribution after serving all patients in this slot
            # First convolve with the service time distribution x times
            sp = np.convolve(sp, convolutions.get(x))
            
            # Then adjust for the duration threshold d
            if len(sp) > d:
                sp[d] = np.sum(sp[:d + 1])
                sp = sp[d:]
            else:
                # If sp is shorter than d, create a new array with one element (sum of all probabilities)
                sp = np.array([np.sum(sp)], dtype=np.float64)
    
    # Expected spillover time at the end
    esp = np.dot(range(len(sp)), sp)
    
    return ewt, esp
