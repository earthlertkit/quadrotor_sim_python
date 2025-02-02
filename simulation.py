import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import quaternion_math as qt
import sensor_data
import path_planning
import position_controller
import attitude_controller
import dynamics
import motor_model
import plotter
import time as timing

def run_simulation():

    # Initial Condition
    state_current = np.zeros(13)
    state_current[6] = 1

    # Simulation parameters
    dt = 0.01

    # Quadrotor parameters
    quadrotor_params = {
        "mass": 1e-1,
        "gravity": np.array([0, 0, -9.81]),
        "moment_of_inertia": np.eye(3) * 1e-4,
        "thrust_coeff": 1e-5,
        "moment_scale": 1e-10,
        "moment_arm_length": 1e-1,
        "motor_constant": 36.5
    }
    
    # Sensor parameters

    # Controller parameters 
    Kp = np.array([1, 1, 5]) 
    Ki = np.array([0, 0, 0])
    Kd = np.array([1, 1, 4])
    Kq = 0
    Kw = 0

    # Initializing quadrotor
    quadrotor = dynamics.Quadrotor(state_current, quadrotor_params, dt)

    # Initializing controllers
    pos_controller = position_controller.PID(Kp, Ki, Kd)
    att_controller = attitude_controller.P2(Kq, Kw)

    # Initializing motors
    motor = motor_model.Motor(quadrotor_params)

    # Generating path from waypoints [x, y, z, yaw]
    waypoints = np.array([[0, 0, 0, 0],
                         [0, 0, 10, 0]]).T
    waypoint_times = np.array([0, 10])
    path_desired = path_planning.waypoint_discretize(waypoints=waypoints, waypoint_times=waypoint_times, dt=dt)

    # Plotting variables
    time = []
    state_current_plot = np.zeros((13, path_desired.shape[1]))
    state_desired_plot = np.zeros((13, path_desired.shape[1]))
    acc_IMU_plot = np.zeros((3, path_desired.shape[1]))
    omega_IMU_plot = np.zeros((3, path_desired.shape[1]))
    rpm_plot = np.zeros((4, path_desired.shape[1]))

    # Control loop
    s = timing.time()
    for i in range(path_desired.shape[1]):
        
        # Get sensor data
        state_current = quadrotor.get_state()
        acc_IMU, omega_IMU = quadrotor.get_IMU_data()
        
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
        # Motor model
        thrust_motor, torque_motor = motor.control(thrust_req, torque_req)

        # Plotting
        time.append(i*dt)
        state_current_plot[:, i] = state_current
        state_desired_plot[:, i] = state_desired
        acc_IMU_plot[:, i] = acc_IMU
        omega_IMU_plot[:, i] = omega_IMU
        rpm_plot[:, i] = motor.get_rpm_data()

        # Updates
        motor.update(dt)
        quadrotor.set_thrust(thrust_motor)
        quadrotor.set_torque(torque_motor)
        quadrotor.update()

        e = timing.time()
        print(e-s)
        

    # Outputting plots
    plotter.position_plot(time, state_current_plot[0:3], state_desired_plot[0:3])
    plotter.orientation_plot(time, state_current_plot[6:10], state_desired_plot[6:10])
    plotter.gyroscope_plot(time, omega_IMU_plot)
    plotter.accelerometer_plot(time, acc_IMU_plot)
    plotter.motor_plot(time, rpm_plot)
    plt.show()

if __name__ == "__main__":
    run_simulation()