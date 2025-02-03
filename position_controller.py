import numpy as np

class PID:
    def __init__(self, Kp, Ki, Kd):
        # Kp, Ki, and Kd are arrays of the form [x gain, y gain, z gain]
        self.Kp = Kp
        self.Ki = Ki
        self.Kd=  Kd
        self.e_i = np.array([0., 0., 0.]).T

    def control(self, state_current, state_desired, dt, params):
        # Proportional (position) error
        e_p = state_desired[0:3] - state_current[0:3]

        # Derivative (velocity) error
        e_d = state_desired[3:6] - state_current[3:6]

        # Integral error
        self.e_i += e_p * dt

        # Acceleration required
        x_acc = self.Kp[0] * e_p[0] + self.Ki[0] * self.e_i[0] + self.Kd[0] * e_d[0]
        y_acc = self.Kp[1] * e_p[1] + self.Ki[1] * self.e_i[1] + self.Kd[1] * e_d[1]
        z_acc = self.Kp[2] * e_p[2] + self.Ki[2] * self.e_i[2] + self.Kd[2] * e_d[2]
        acc = np.array([x_acc, y_acc, z_acc]) - params["gravity"]

        # Thrust required
        T = params["mass"] * np.linalg.norm(acc)
        
        return T, acc