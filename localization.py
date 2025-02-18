import numpy as np
import quaternion_math as qt

class EKF:
    def __init__(self, state_ekf_init, P, Q, R, params, dt):
        self.state_ekf = state_ekf_init
        self.P = P
        self.Q = Q
        self.R = R
        self.a_IMU = np.zeros(3)
        self.w_IMU = np.zeros(3)
        self.gps_data = np.zeros(3)
        self.vel_data = np.zeros(3)
        self.quat_data = np.zeros(4)
        self.params = params
        self.dt = dt
    
    
    def state_dot(self, x):

        q = x[0:4]
        v = x[7:10]

        q_dot = -0.5 * qt.multiply(np.array([0, *self.w_IMU]), q)
        r_dot = v
        v_dot = qt.quat2rot(q) @ self.a_IMU + self.params["gravity"]
        b_a_dot = np.zeros(3)
        b_w_dot = np.zeros(3)
    
        return np.array([*q_dot, *r_dot, *v_dot, *b_a_dot, *b_w_dot])
    

    def compute_F(self, x):
        a_IMU = self.a_IMU
        w_IMU = self.w_IMU
        q = x[0:4]

        F = np.zeros((16, 16))
        F[0:4, 0:4] = 0.5 * np.array([[0, -w_IMU[0], -w_IMU[1], -w_IMU[2]],
                                    [w_IMU[0], 0, w_IMU[2], -w_IMU[1]],
                                    [w_IMU[1], -w_IMU[2], 0,w_IMU[0]],
                                    [w_IMU[2], w_IMU[1], -w_IMU[0], 0]])
        F[0:4, 13:16] = 0.5 * np.array([[q[1], q[2], q[3]],
                                        [-q[0], q[3], -q[2]],
                                        [-q[3], -q[0], q[1]],
                                        [q[2], -q[1], -q[0]]])
        F[4:7, 7:10] = np.eye(3)
        F[7:10, 0:4] = np.array([[-2*(q[3]*a_IMU[1]-q[2]*a_IMU[2]), 2*(q[2]*a_IMU[1]+q[3]*a_IMU[2]), -2*(2*q[2]*a_IMU[0]-q[1]*a_IMU[1]-q[0]*a_IMU[2]), -2*(2*q[3]*a_IMU[0]+q[0]*a_IMU[1]-q[1]*a_IMU[2])],
                                [-2*(q[1]*a_IMU[2]-q[3]*a_IMU[0]), -2*(2*q[1]*a_IMU[1]-q[2]*a_IMU[0]+q[0]*a_IMU[2]), 2*(q[1]*a_IMU[0]+q[3]*a_IMU[2]), -2*(2*q[3]*a_IMU[1]-q[0]*a_IMU[0]-q[2]*a_IMU[2])],
                                [-2*(q[2]*a_IMU[0]-q[1]*a_IMU[1]), -2*(2*q[1]*a_IMU[2]-q[3]*a_IMU[0]-q[0]*a_IMU[1]), -2*(2*q[2]*a_IMU[2]-q[3]*a_IMU[1]+q[0]*a_IMU[0]), 2*(q[1]*a_IMU[0]+q[2]*a_IMU[1])]]) 
        F[7:10, 10:13] = np.array([[-1+2*(q[2]**2+q[3]**2), -2*(q[1]*q[2]-q[0]*q[3]), -2*(q[1]*q[3]+q[0]*q[2])],
                                [-2*(q[0]*q[3]+q[1]*q[2]), -1+2*(q[1]**2+q[3]**2), -2*(q[2]*q[3]+q[0]*q[1])],
                                [-2*(q[1]*q[3]-q[0]*q[2]), -2*(q[0]*q[1]+q[2]*q[3]), -1+2*(q[1]**2+q[2]**2)]])

        return F
    

    def compute_H(self):

        H = np.zeros((10, 16))
        H[0:10, 0:10] = np.eye(10)

        return H
    

    def prediction_step(self):

        # Improved Euler's Method for predicting next time step
        x_plus = self.state_ekf
        x_dot = self.state_dot(x_plus)
        x_dot_minus = self.state_dot(x_plus + self.dt * x_dot)
        x_minus = x_plus + self.dt/2 * (x_dot + x_dot_minus)

        # Updating Error Covariance Matrix
        P_plus = self.P
        F = self.compute_F(x_minus)
        P_dot = F @ P_plus + P_plus @ F.T + self.Q
        P_minus = P_plus + self.dt * P_dot

        return x_minus, P_minus
    

    def measurement_update(self, x_minus, P_minus):

        # Computing Kalman gain
        H = self.compute_H()
        S = H @ P_minus @ H.T + self.R
        S_inv = np.linalg.solve(S, np.eye(10))
        K = P_minus @ H.T @ S_inv

        # Updating prediction
        x = x_minus + K @ (np.array([*self.quat_data, *self.gps_data, *self.vel_data]) - x_minus[0:10])
        P = (np.eye(16) - K @ H) @ P_minus

        return x, P

    
    def update(self):

        x_minus, P_minus = self.prediction_step()
        x, P = self.measurement_update(x_minus, P_minus)

        self.state_ekf = x
        self.P = P


    def update_sensor_data(self, a_IMU, w_IMU, gps_data, vel_data, quat_data):

        self.a_IMU = a_IMU
        self.w_IMU = w_IMU
        self.gps_data = gps_data
        self.vel_data = vel_data
        self.quat_data = quat_data


    def get_state(self):
        return self.state_ekf