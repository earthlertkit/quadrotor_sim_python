import numpy as np
from scipy.integrate import solve_ivp
import quaternion_math as qt

class Quadrotor:
    def __init__(self, state_init, params, dt):

        self.state = state_init
        self.params = params
        self.dt = dt
        self.thrust = 0
        self.torque = np.zeros(3)

    
    def quadrotor_dynamics(self, t, x):

        m = self.params["mass"]
        g = self.params["gravity"]
        I = self.params["moment_of_inertia"]    

        v = x[3:6]
        q = x[6:10]
        w = x[10:13]

        r_dot = v
        v_dot = qt.quat2rot(q) @ np.array([0, 0, self.thrust / m]) + g
        q_dot = -0.5 * qt.multiply(np.array([0, *w]), q)
        w_dot = np.linalg.solve(I, self.torque - np.cross(w, I @ w))

        return np.array([*r_dot, *v_dot, *q_dot, *w_dot])
    

    def update(self):
        
        # Quadrotor dynamics solver
        solution = solve_ivp(
            self.quadrotor_dynamics,
            (0.0, self.dt),
            self.state,
            t_eval=[self.dt],
            method="RK45"
        )

        # Update
        self.state = solution.y[:, -1]
        self.state[6:10] /= np.linalg.norm(self.state[6:10])


    def get_IMU_data(self):

        # Simulate dynamics
        x_dot = self.quadrotor_dynamics(None, self.state)

        # Accelerometer data
        a_IMU = qt.quat2rot(self.state[6:10]).T @ (x_dot[3:6] - self.params["gravity"]) + np.random.normal(0, 0.01, 3)
        
        # Gyro data
        w_IMU = self.state[10:13] + np.random.normal(0, 0.01, 3)

        return a_IMU, w_IMU

    
    def get_GPS_data(self):
        return self.state[0:3] + np.random.normal(0, 0.1, 3)
    

    def get_vel_data(self):
        return self.state[3:6] + np.random.normal(0, 0.1, 3)
    

    def get_quat_data(self):
        return self.state[6:10] + np.random.normal(0, 0.01, 4)


    def get_state(self):
        return self.state
    

    def set_thrust(self, thrust):
        self.thrust = thrust

    
    def set_torque(self, torque):
        self.torque = torque