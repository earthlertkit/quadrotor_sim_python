import numpy as np
from scipy.integrate import solve_ivp

class Gyroscope:
    def __init__(self, omega_init, measurement_rate):
        self.omega = omega_init
        self.dt = measurement_rate

    def rotational_dynamics(self, torque, params):
        I = params["MomentOfInertia"]
        omega_dot = np.linalg.solve(I, torque - np.cross(self.omega, I @ self.omega))

        return omega_dot
        
    def update(self, torque, params):
        pass