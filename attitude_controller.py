import numpy as np
import quaternion_math as qt
import sensor_data

class P2:
    def __init__(self, Kq, Kw):
        # Kq and Kw are scalars corresponding to quaternion and angular velocity gains respectively
        self.Kq = Kq
        self.Kw = Kw 

    def control(self, state_current, state_desired, params, acc, omega):
        # Quaternion error
        q_desired = self.desired_quaternions(state_current, state_desired, params, acc)
        q_current = state_current[6:10]
        q_error = qt.multiply(q_desired, qt.conjugate(q_current))
        q_error /= np.linalg.norm(q_error)
        q_error_vec = q_error[1:4] * np.sign(q_error[0])

        # Required torque
        torque = -self.Kq * q_error_vec - self.Kw * omega

        # New desired state with updated attitudes
        state_desired_new = state_desired.copy()
        state_desired_new[6:10] = q_desired

        return torque, state_desired_new

    def desired_quaternions(self, state_current, state_desired, params, acc):
        # Yaw desired
        yaw_desired = qt.quat2eul(state_desired[6:10])[2]

        # Computing DCM
        z_b = acc
        z_b /= np.linalg.norm(z_b)

        x_c = np.array([np.cos(yaw_desired), np.sin(yaw_desired), 0])

        y_b = np.cross(z_b, x_c) / np.linalg.norm(np.cross(z_b, x_c))

        x_b = np.cross(y_b, z_b)

        R = np.array([x_b, y_b, z_b])
        
        q_desired = qt.rot2quat(R)
        q_desired /= np.linalg.norm(q_desired)
        
        return q_desired

