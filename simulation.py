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
    Kp = np.array([0, 0, 5])
    Ki = np.array([0, 0, 0])
    Kd = np.array([0, 0, 4])
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
                         [0, 0, 0, 0]]).T
    waypoint_times = np.array([0, 5, 10])
    path_desired = path_planning.waypoint_discretize(waypoints=waypoints, waypoint_times=waypoint_times, dt=dt)

    # Plotting variables
    time = []
    z_pos = []
    z_pos_des = []

    # Control loop
    for i in range(path_desired.shape[1]):
        
        # Plotting
        time.append(i*dt)
        z_pos.append(state_current[6])
        z_pos_des.append(path_desired[6, i])

        # Get sensor data
        acc_IMU = accelerometer.get_sensor_data()
        omega_IMU = gyroscope.get_sensor_data()
        
        # Position controller
        thrust_req, acc_req = pos_controller.control(state_current=state_current, 
                                                     state_desired=path_desired[:, i], 
                                                     dt=dt, 
                                                     params=quadrotor_params)

        # Attitude controller
        torque_req = att_controller.control(state_current=state_current,
                                            state_desired=path_desired[:, i],
                                            params=quadrotor_params,
                                            acc=acc_req,
                                            omega=omega_IMU)
        
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

    plt.plot(time, z_pos, label="actual")
    plt.plot(time, z_pos_des, label="desired")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_simulation()