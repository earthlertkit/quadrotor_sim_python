import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import quaternion_math as qt
import sensor_data
import path_planning
import position_controller
import attitude_controller
import dynamics


def run_simulation():

    # Initial Condition
    state_current = np.zeros(16)

    # Simulation parameters
    dt = 0.01

    # Quadrotor parameters
    quadrotor_params = {
        "mass": 1.0,
        "gravity": np.array([0, 0, -9.81]),
        "moment_of_inertia": np.eye(3)
    }

    # Sensor parameters

    # Controller parameters
    Kp = np.array([5, 5, 5])
    Ki = np.array([0, 0, 0])
    Kd = np.array([4, 4, 4])
    Kq = 0
    Kw = 0

    # Initializing sensors
    accelerometer = sensor_data.Accelerometer(np.array([0, 0, 0]))
    gyroscope = sensor_data.Gyroscope(np.array([0, 0, 0]), dt)

    # Initializing controllers
    pos_controller = position_controller.PID(Kp, Ki, Kd)
    att_controller = attitude_controller.P2(Kq, Kw)

    # Generating path from waypoints [x, y, z, yaw]
    waypoints = np.array([[0, 0, 0, 0],
                         [0, 0, 10, 0],
                         [10, 0, 10, 0],
                         [10, 10, 10, 0],
                         [10, 10, 10, np.pi],
                         [0, 0, 0, 0]]).T
    waypoint_times = np.array([0, 5, 10, 15, 20, 25])
    path_desired = path_planning.waypoint_discretize(waypoints=waypoints, waypoint_times=waypoint_times, dt=dt)

    # Plotting variables
    time = []
    state_current_plot = np.zeros((16, path_desired.shape[1]))
    state_desired_plot = np.zeros((16, path_desired.shape[1]))

    # Control loop
    for i in range(path_desired.shape[1]):

        # Get sensor data
        acc_IMU = accelerometer.get_sensor_data()
        omega_IMU = gyroscope.get_sensor_data()
        
        # Position controller
        thrust_req, acc_req = pos_controller.control(state_current=state_current, 
                                                     state_desired=path_desired[:, i], 
                                                     dt=dt, 
                                                     params=quadrotor_params)

        # Attitude controller
        torque_req, state_desired = att_controller.control(state_current=state_current,
                                            state_desired=path_desired[:, i],
                                            params=quadrotor_params,
                                            acc=acc_req,
                                            omega=omega_IMU)
        
        # Plotting
        time.append(i*dt)
        state_current_plot[:, i] = state_current
        state_desired_plot[:, i] = state_desired

        # Simulating dynamics
        solution = solve_ivp(
            dynamics.quadrotor_dynamics,
            (0.0, dt),
            state_current,
            args=(omega_IMU, acc_IMU, quadrotor_params),
            t_eval=np.linspace(0.0, dt, 10),
            method="RK45"
        )
        state_current = solution.y[:, -1]

        # Update sensors
        gyroscope.update(torque_req, quadrotor_params)
        accelerometer.update(state_current, acc_req, quadrotor_params)

    # Outputting position plots
    fig, ax = plt.subplots(3, 1)

    ax[0].plot(time, state_current_plot[4, :], label='Actual')
    ax[0].plot(time, state_desired_plot[4, :], label='Desired')
    ax[0].set_title("x")
    ax[0].legend()

    ax[1].plot(time, state_current_plot[5, :], label='Actual')
    ax[1].plot(time, state_desired_plot[5, :], label='Desired')
    ax[1].set_title("y")
    ax[1].legend()

    ax[2].plot(time, state_current_plot[6, :], label='Actual')
    ax[2].plot(time, state_desired_plot[6, :], label='Desired')
    ax[2].set_title("x")
    ax[2].legend()

    plt.show()

if __name__ == "__main__":
    run_simulation()