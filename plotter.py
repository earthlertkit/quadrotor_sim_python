import numpy as np
import matplotlib.pyplot as plt
import quaternion_math as qt

def position_plot(t, r_actual, r_desired, r_ekf):
    fig, ax = plt.subplots(3, 1)

    ax[0].plot(t, r_actual[0], label='Actual')
    ax[0].plot(t, r_desired[0], label='Desired')
    ax[0].plot(t, r_ekf[0], label='EKF')
    ax[0].set_ylabel("x (m)")
    ax[0].set_xlabel("t (s)")
    ax[0].legend()
    ax[0].legend(loc='lower right')

    ax[1].plot(t, r_actual[1], label='Actual')
    ax[1].plot(t, r_desired[1], label='Desired')
    ax[1].plot(t, r_ekf[1], label='EKF')
    ax[1].set_ylabel("y (m)")
    ax[1].set_xlabel("t (s)")
    ax[1].legend()
    ax[1].legend(loc='lower right')

    ax[2].plot(t, r_actual[2], label='Actual')
    ax[2].plot(t, r_desired[2], label='Desired')
    ax[2].plot(t, r_ekf[2], label='EKF')
    ax[2].set_ylabel("z (m)")
    ax[2].set_xlabel("t (s)")
    ax[2].legend()
    ax[2].legend(loc='lower right')


def velocity_plot(t, v_actual, v_desired, v_ekf):
    fig, ax = plt.subplots(3, 1)

    ax[0].plot(t, v_actual[0], label='Actual')
    ax[0].plot(t, v_desired[0], label='Desired')
    ax[0].plot(t, v_ekf[0], label='EKF')
    ax[0].set_title("u")
    ax[0].legend()

    ax[1].plot(t, v_actual[1], label='Actual')
    ax[1].plot(t, v_desired[1], label='Desired')
    ax[1].plot(t, v_ekf[1], label='EKF')
    ax[1].set_title("v")
    ax[1].legend()

    ax[2].plot(t, v_actual[2], label='Actual')
    ax[2].plot(t, v_desired[2], label='Desired')
    ax[2].plot(t, v_ekf[2], label='EKF')
    ax[2].set_title("w")
    ax[2].legend()


def orientation_plot(t, q_actual, q_desired, q_ekf):
    angles_current_plot = np.zeros((3, len(t)))
    angles_desired_plot = np.zeros((3, len(t)))
    angles_ekf_plot = np.zeros((3, len(t)))

    for j in range(len(t)):
        angles_current_plot[:, j] = qt.quat2eul(q_actual[:, j])
        angles_desired_plot[:, j] = qt.quat2eul(q_desired[:, j])
        angles_ekf_plot[:, j] = qt.quat2eul(q_ekf[:, j])

    fig, ax = plt.subplots(3, 1)

    ax[0].plot(t, angles_current_plot[0], label='Actual')
    ax[0].plot(t, angles_desired_plot[0], label='Desired')
    ax[0].plot(t, angles_ekf_plot[0], label='EKF')
    ax[0].set_ylabel("roll (rad)")
    ax[0].set_xlabel("t (s)")
    ax[0].legend()
    ax[0].legend(loc='lower right')

    ax[1].plot(t, angles_current_plot[1], label='Actual')
    ax[1].plot(t, angles_desired_plot[1], label='Desired')
    ax[1].plot(t, angles_ekf_plot[1], label='EKF')
    ax[1].set_ylabel("pitch (rad)")
    ax[1].set_xlabel("t (s)")
    ax[1].legend()
    ax[1].legend(loc='lower right')

    ax[2].plot(t, angles_current_plot[2], label='Actual')
    ax[2].plot(t, angles_desired_plot[2], label='Desired')
    ax[2].plot(t, angles_ekf_plot[2], label='EKF')
    ax[2].set_ylabel("yaw (rad)")
    ax[2].set_xlabel("t (s)")
    ax[2].legend()
    ax[2].legend(loc='lower right')


def gyroscope_plot(t, omega):
    fig, ax = plt.subplots(3, 1)

    ax[0].plot(t, omega[0])
    ax[0].set_title("angular velocity in x")

    ax[1].plot(t, omega[1])
    ax[1].set_title("angular velocity in y")

    ax[2].plot(t, omega[2])
    ax[2].set_title("angular velocity in z")


def accelerometer_plot(t, acc):
    fig, ax = plt.subplots(3, 1)

    ax[0].plot(t, acc[0])
    ax[0].set_title("acceleration in x")

    ax[1].plot(t, acc[1])
    ax[1].set_title("acceleration in y")

    ax[2].plot(t, acc[2])
    ax[2].set_title("acceleration in z")


def motor_plot(t, rpm_current, rpm_desired):
    fig, ax = plt.subplots(4, 1)

    ax[0].plot(t, rpm_current[0], label="Actual")
    ax[0].plot(t, rpm_desired[0], label="Desired")
    ax[0].set_title("motor 1 rpm")
    ax[0].legend()

    ax[1].plot(t, rpm_current[1], label="Actual")
    ax[1].plot(t, rpm_desired[1], label="Desired")
    ax[1].set_title("motor 2 rpm")
    ax[1].legend()

    ax[2].plot(t, rpm_current[2], label="Actual")
    ax[2].plot(t, rpm_desired[2], label="Desired")
    ax[2].set_title("motor 3 rpm")
    ax[2].legend()

    ax[3].plot(t, rpm_current[3], label="Actual")
    ax[3].plot(t, rpm_desired[3], label="Desired")
    ax[3].set_title("motor 4 rpm")
    ax[3].legend()