import numpy as np
import quaternion_math as qt

def rotational_dynamics(t, omega, torque, params):
    I = params["MomentOfInertia"]
    omega_dot = np.linalg.solve(I, torque - np.cross(omega, I @ omega))

    return omega_dot

def quadrotor_dynamics(t, x, omega, acc, params):
    q = x[0:4]
    v = x[7:10]

    # Quaternions
    q_dot = 0.5 * np.array([[0, -omega[0], -omega[1], -omega[2]],
                            [omega[0], 0, omega[2], -omega[1]],
                            [omega[1], -omega[2], 0, omega[0]],
                            [omega[2], omega[1], -omega[0], 0]]) @ q

    # Positions
    r_dot = v

    # Velocities
    v_dot = qt.quat2rot(q).T @ acc + params["gravity"]

    # Accelerometer biases
    b_a_dot = np.zeros(3)


    # Gyroscope biases
    b_w_dot = np.zeros(3)

    return np.concatenate((q_dot, r_dot, v_dot, b_a_dot, b_w_dot))