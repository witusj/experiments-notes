from typing import List

def generate_evenly_distributed_schedule_intervals(n_patients: int, t_intervals: int) -> List[int]:
    """
    Generates a list representing a schedule, distributing 'n_patients'
    into 't_intervals' as evenly as possible with guaranteed placement.
    A '1' indicates a scheduled interval, and '0' an empty one.
    
    Key Behavior:
    - If n_patients == 0: Returns a list of all zeros of length 't_intervals'.
    - If t_intervals < n_patients (and n_patients > 0): Raises a ValueError.
    - If n_patients == t_intervals (and both > 0): Returns a list of all ones.
    - If n_patients < t_intervals (and both > 0): Places each patient at the 
      center of their allocated segment to ensure even distribution without collisions.
    
    **Guarantee**: This approach ensures exactly 'n_patients' will be scheduled
    with no duplicates or collisions.
    
    Args:
        n_patients (int): The number of patients to schedule.
                          Must be a non-negative integer.
        t_intervals (int): The total number of available time intervals.
                           Must be a non-negative integer.
    
    Returns:
        List[int]: A list of integers (0s and 1s) of length 't_intervals',
                   representing the schedule. '1' for a scheduled patient, '0' otherwise.
    
    Raises:
        ValueError: If 'n_patients' or 't_intervals' are not non-negative integers.
        ValueError: If 't_intervals' is less than 'n_patients' (and n_patients > 0),
                    as it's impossible to schedule more patients than available intervals.
    """
    # Validate inputs
    if not isinstance(n_patients, int) or n_patients < 0:
        raise ValueError("n_patients must be a non-negative integer.")
    if not isinstance(t_intervals, int) or t_intervals < 0:
        raise ValueError("t_intervals must be a non-negative integer.")
    
    # Handle n_patients == 0
    if n_patients == 0:
        return [0] * t_intervals
    
    # At this point, n_patients > 0
    if t_intervals < n_patients:
        raise ValueError(
            f"Cannot schedule {n_patients} patients in only {t_intervals} intervals. "
            "Not enough unique slots available."
        )
    elif n_patients == t_intervals:
        return [1] * t_intervals
    else:  # n_patients < t_intervals
        schedule = [0] * t_intervals
        
        # Place each patient at the center of their allocated segment
        # This guarantees no collisions and even distribution
        for i in range(n_patients):
            # Each patient gets a segment of size (t_intervals / n_patients)
            # Place patient at center: (i + 0.5) * segment_size
            position = int((i + 0.5) * t_intervals / n_patients)
            schedule[position] = 1
        
        return schedule


# Alternative approach using Bresenham-like algorithm
def generate_evenly_distributed_schedule_intervals_bresenham(n_patients: int, t_intervals: int) -> List[int]:
    """
    Alternative implementation using a Bresenham-like algorithm for even distribution.
    This approach also guarantees all patients are scheduled.
    """
    # Same validation as above
    if not isinstance(n_patients, int) or n_patients < 0:
        raise ValueError("n_patients must be a non-negative integer.")
    if not isinstance(t_intervals, int) or t_intervals < 0:
        raise ValueError("t_intervals must be a non-negative integer.")
    
    if n_patients == 0:
        return [0] * t_intervals
    
    if t_intervals < n_patients:
        raise ValueError(
            f"Cannot schedule {n_patients} patients in only {t_intervals} intervals. "
            "Not enough unique slots available."
        )
    elif n_patients == t_intervals:
        return [1] * t_intervals
    else:
        schedule = [0] * t_intervals
        
        # Bresenham-like algorithm for even distribution
        error = t_intervals // 2
        patients_scheduled = 0
        
        for i in range(t_intervals):
            error += n_patients
            if error >= t_intervals:
                schedule[i] = 1
                patients_scheduled += 1
                error -= t_intervals
                
                # Stop once we've scheduled all patients
                if patients_scheduled >= n_patients:
                    break
        
        return schedule


# Test function to verify both approaches
def test_scheduling_functions():
    """Test both scheduling approaches with various inputs."""
    test_cases = [
        (0, 5),   # No patients
        (3, 3),   # Same number
        (3, 10),  # Even distribution
        (5, 12),  # Different ratio
        (2, 7),   # Small numbers
        (1, 5),   # Single patient
    ]
    
    print("Testing scheduling functions:")
    print("=" * 50)
    
    for n_patients, t_intervals in test_cases:
        print(f"\nTest: {n_patients} patients, {t_intervals} intervals")
        
        # Test centered approach
        result1 = generate_evenly_distributed_schedule_intervals(n_patients, t_intervals)
        scheduled1 = sum(result1)
        
        # Test Bresenham approach
        result2 = generate_evenly_distributed_schedule_intervals_bresenham(n_patients, t_intervals)
        scheduled2 = sum(result2)
        
        print(f"Centered:   {result1} (scheduled: {scheduled1})")
        print(f"Bresenham:  {result2} (scheduled: {scheduled2})")
        print(f"Expected:   {n_patients} patients scheduled")
        print(f"Success:    {scheduled1 == n_patients and scheduled2 == n_patients}")

if __name__ == "__main__":
    test_scheduling_functions()

