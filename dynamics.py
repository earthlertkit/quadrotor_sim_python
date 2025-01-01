import numpy as np

def rotational_dynamics(t, omega, torque, params):
    I = params["MomentOfInertia"]
    omega_dot = np.linalg.solve(I, torque - np.cross(omega, I @ omega))

    return omega_dot