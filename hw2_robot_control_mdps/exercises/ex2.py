import numpy as np


def generate_quintic_spline_waypoints(start, end, num_points):

    """
    TODO:

    Steps:
    1. Generate `num_points` linearly spaced time steps `s` between 0 and 1.
    2. Apply the quintic time scaling polynomial function which can be found in the slides to get `f_s`.
    3. Interpolate between `start` and `end` using `start + (end - start) * f_s`.
    
    Args:
        start (np.ndarray): Starting waypoint.
        end (np.ndarray): Ending waypoint.
        num_points (int): Number of points in the trajectory.
        
    Returns:
        np.ndarray: Generated waypoints.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    # normalized time s in [0, 1]
    s = np.linspace(0.0, 1.0, num_points)
    # quintic time scaling: f(0)=0, f(1)=1, and zero velocity AND acceleration at both ends -> smooth start and stop
    f_s = 10 * s**3 - 15 * s**4 + 6 * s**5
    # broadcast f_s (num_points,) against the waypoint dimensions, e.g. (num_points, 1) * (3,) -> (num_points, 3)
    f_s = f_s.reshape(-1, *([1] * start.ndim))
    return start + (end - start) * f_s


def pid_control(tracking_error_history, timestep, Kp=150.0, Ki=0.0, Kd=0.01):
    """
    TODO:
    Compute the PID control signal based on the tracking error history.
    
    Steps:
    1. The Proportional (P) term is the most recent error.
    2. The Integral (I) term is the sum of all past errors, multiplied by the simulation timestep.
    3. The Derivative (D) term is the rate of change of the error (difference between the last two errors divided by the timestep).
       If there is only one error in history, the D term should be zero.
    4. Compute the final control signal: Kp * P + Ki * I + Kd * D.
    
    Args:
        tracking_error_history (np.ndarray): History of tracking errors.
        timestep (float): Simulation timestep.
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        
    Returns:
        np.ndarray: Control signal.
    """
    errors = np.atleast_2d(np.asarray(tracking_error_history, dtype=float))   # (T, num_joints)
    # P: current error
    P = errors[-1]
    # I: accumulated error over the history, approximated as a Riemann sum
    I = errors.sum(axis=0) * timestep
    # D: finite-difference rate of change of the error; zero when only one sample exists
    if len(errors) > 1:
        D = (errors[-1] - errors[-2]) / timestep
    else:
        D = np.zeros_like(P)
    return Kp * P + Ki * I + Kd * D
            