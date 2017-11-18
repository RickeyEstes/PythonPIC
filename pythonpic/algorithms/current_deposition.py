# coding=utf-8
import numpy as np
# from numba import jit # TODO

def aperiodic_current_deposition(j_x, j_yz, velocity, x_particles, dx, dt, q):
    current_deposition(j_x, j_yz, velocity, x_particles, dx, dt, q)
    j_yz[:2] = 0
    j_yz[-2:] = 0
    j_x[0] = 0
    j_x[-2:] = 0

# @jit()
def periodic_current_deposition(j_x, j_yz, velocity, x_particles, dx, dt, q):
    current_deposition(j_x, j_yz, velocity, x_particles, dx, dt, q)
    j_yz[-4:-2] += j_yz[:2]
    j_yz[2:4] += j_yz[-2:]
    j_x[-3] += j_x[0]
    j_x[0] = 0
    j_x[1:3] += j_x[-2:]
    j_x[-2:] = 0

def current_deposition(j_x, j_yz, velocity, x_particles, dx, dt, q):
    epsilon = dx * 1e-10
    time = np.ones_like(x_particles) * dt
    active = np.any(velocity,axis=1)

    while active.any():
        logical_coordinates_n = (x_particles // dx).astype(np.int32)
        particle_in_left_half = x_particles / dx - logical_coordinates_n < 0.5
        particle_in_right_half = ~particle_in_left_half
        x_velocity = velocity[:, 0]

        velocity_to_left = x_velocity < 0
        velocity_to_right = x_velocity > 0
        velocity_zero = x_velocity == 0

        t1 = np.empty_like(x_particles)
        s = np.empty_like(x_particles)

        case1 = particle_in_left_half & velocity_to_left
        case2 = particle_in_left_half & velocity_to_right
        case3 = particle_in_right_half & velocity_to_right
        case4 = particle_in_right_half & velocity_to_left

        t1[case1] = - (x_particles[case1] - logical_coordinates_n[case1] * dx) / x_velocity[case1]
        t1[case2] = ((logical_coordinates_n[case2] + 0.5) * dx - x_particles[case2]) / x_velocity[case2]
        t1[case3] = ((logical_coordinates_n[case3] + 1) * dx - x_particles[case3]) / x_velocity[case3]
        t1[case4] = -(x_particles[case4] - (logical_coordinates_n[case4] + 0.5) * dx) / x_velocity[case4]
        t1[velocity_zero] = np.inf

        s[case1] = logical_coordinates_n[case1] * dx - epsilon
        s[case2] = (logical_coordinates_n[case2] + 0.5) * dx + epsilon
        s[case3] = (logical_coordinates_n[case3] + 1) * dx + epsilon
        s[case4] = (logical_coordinates_n[case4] + 0.5) * dx - epsilon
        s[velocity_zero] = x_particles[velocity_zero]

        time_overflow = time - t1
        switches_cells = time_overflow > 0
        time_in_this_iteration = np.where(switches_cells, t1, time)
        time_in_this_iteration[x_velocity == 0] = dt

        logical_coordinates_long = np.where(particle_in_right_half, logical_coordinates_n+1, logical_coordinates_n)
        logical_coordinates_trans = np.where(particle_in_left_half, logical_coordinates_n-1, logical_coordinates_n +1)

        sign = particle_in_left_half.astype(int) * 2 - 1
        distance_to_current_cell_center = (logical_coordinates_n + 0.5) * dx - x_particles
        s0 = 1 - sign * distance_to_current_cell_center / dx
        change_in_coverage = sign * x_velocity * time_in_this_iteration / dx
        s1 = s0 + change_in_coverage
        w = 0.5 * (s0 + s1)

        j_contribution = velocity * q / dt * time_in_this_iteration.reshape(
            x_velocity.size, 1)
        y_contribution_to_current_cell = w * j_contribution[:,1]
        z_contribution_to_current_cell = w * j_contribution[:,2]
        y_contribution_to_next_cell = (1 - w) * j_contribution[:,1]
        z_contribution_to_next_cell = (1 - w) * j_contribution[:,2]

        j_x += np.bincount(logical_coordinates_long + 1, j_contribution[:,0],
                           minlength=j_x.size)
        j_yz[:, 0] += np.bincount(logical_coordinates_n + 2, y_contribution_to_current_cell, minlength=j_yz[:, 1].size)
        j_yz[:, 1] += np.bincount(logical_coordinates_n + 2, z_contribution_to_current_cell, minlength=j_yz[:, 1].size)

        j_yz[:, 0] += np.bincount(logical_coordinates_trans + 2, y_contribution_to_next_cell, minlength=j_yz[:, 1].size)
        j_yz[:, 1] += np.bincount(logical_coordinates_trans + 2, z_contribution_to_next_cell, minlength=j_yz[:, 1].size)

        time = time_overflow[switches_cells]
        x_particles = s[switches_cells]
        velocity = velocity[switches_cells]
        active = np.ones_like(x_particles, dtype=bool)
