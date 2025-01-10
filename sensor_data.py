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
            t_eval=np.linspace(0.0, self.dt, 10),
            method="RK45"
        )

        # Update
        self.omega = solution.y[:, -1]

    def get_sensor_data(self):
        return self.omega
    
    
class Accelerometer:
    def __init__(self, acc_init):
        self.acc = acc_init

    def update(self, state_curent, acc_n, params):
        # Currently using desired acceleration straight from position controller
        # Need to update to use thrust produced from motor dynamics instead

        # Converting desired acceleration from inertial to body frame
        R = qt.quat2rot(state_curent[0:4])
        acc_b = R @ (acc_n - params["gravity"])

        # Update
        self.acc = acc_b

    def get_sensor_data(self):
        return self.acc