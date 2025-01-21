import numpy as np
from scipy.integrate import solve_ivp
import dynamics
import quaternion_math as qt

class Gyroscope:
    def __init__(self, omega_init, measurement_rate):
        self.omega = omega_init
        self.dt = measurement_rate
        
    def update(self, torque, params):

        # Rotational dynamics solver
        solution = solve_ivp(
            dynamics.rotational_dynamics,
            (0.0, self.dt),
            self.omega,
            args=(torque, params),
            t_eval=[self.dt],
            method="RK45"
        )

        # Update
        self.omega = solution.y[:, -1]

    def get_sensor_data(self):
        return self.omega
    
    
class Accelerometer:
    def __init__(self, acc_init):
        self.acc = acc_init

    def update(self, state_curent, thrust, params):
        # Update
        self.acc = np.array([0, 0, thrust]) / params["mass"]

    def get_sensor_data(self):
        return self.acc