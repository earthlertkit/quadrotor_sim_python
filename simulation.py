import numpy as np
import matplotlib.pyplot as plt
import quaternion_math as qt
import sensor_data
import path_planning
import position_controller
import attitude_controller
import dynamics


def run_simulation():

    # Simulation parameters
    dt = 0.01

    # Quadrotor parameters

    # Sensor parameters

    # Generating path from waypoints
    waypoints = np.array([[0, 0, 0, 0],
                         [0, 0, 10, 0],
                         [0, 0, 0, 0]]).T
    waypoint_times = np.array([0, 5, 10])
    path_planning.waypoint_discretize(waypoints=waypoints, waypoint_times=waypoint_times, dt=dt)

if __name__ == "__main__":
    run_simulation()