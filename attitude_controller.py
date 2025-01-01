import numpy as np
import quaternion_math as qt

class PD:
    def __init__(self, Kq, Kw):
        # Kq and Kw are scalars corresponding to quaternion and angular velocity gains respectively
        self.Kq = Kq
        self.Kw=  Kw

    def control(self, state_current, state_desired, dt, params, acc):
        # Quaternion error
        q_desired = self.desired_quaternions(state_desired, params, acc)
        q_current = state_current[0:4]
        q_error = qt.multiply(q_desired, qt.conjugate(q_current))[1:4]
    
    def desired_quaternions(self, state_desired, params, acc):
        # Desired z-axis (unit thrust vector)
        T = acc - params["gravity"]
        z_b_desired = T / np.linalg.norm(T)

        # Yaw rotation x-axis
        R_yaw = qt.quat2rot(state_desired[0:4])
        x_b_yaw = R_yaw[:, 0]
        
        # Desired y-axis
        y_b_desired = np.cross(z_b_desired, x_b_yaw)
        y_b_desired /= np.linalg.norm(y_b_desired)

        # Desired x-axis
        x_b_desired = np.cross(y_b_desired, z_b_desired)

        # Desired Rotation Matrix
        R_desired = np.column_stack((x_b_desired, y_b_desired, z_b_desired))

        return qt.rot2quat(R_desired)

