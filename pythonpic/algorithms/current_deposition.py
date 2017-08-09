# coding=utf-8
import numpy as np
# from numba import jit # TODO

c = 299792458 # m /s
# TODO: this can be optimized plenty
# @jit()
def longitudinal_current_deposition(j_x, x_velocity, x_particles, dx, dt, q):
    """

    Parameters
    ----------
    j_x : ndarray
        current in x direction
    x_velocity : ndarray
        x velocity
    x_particles : ndarray
        x position
    dx : float
        grid cell size
    dt : float
        timestep
    q : float
        charge to deposit. for `Species`, this is `eff_q`

    Returns
    -------

    """
    time = np.ones_like(x_particles) * dt
    epsilon = dx * 1e-9
    counter = 0
    # actives = []
    active = x_velocity != 0
    x_particles = x_particles[active]
    x_velocity = x_velocity[active]
    time = time[active]

    while active.any():
        counter += 1
        if counter > 4:
            raise Exception("Infinite recurrence!")
        logical_coordinates_n = (x_particles // dx).astype(np.int32)
        particle_in_left_half = x_particles / dx - logical_coordinates_n <= 0.5
        particle_in_right_half = x_particles / dx - logical_coordinates_n > 0.5 # TODO negate prev`
        velocity_to_left = x_velocity < 0
        velocity_to_right = x_velocity > 0

        case1 = particle_in_left_half & velocity_to_left
        case2 = particle_in_right_half & velocity_to_left
        case3 = particle_in_right_half & velocity_to_right
        case4 = particle_in_left_half & velocity_to_right

        t1 = np.empty_like(x_particles)
        # case1
        t1[case1] = -(x_particles[case1] - (logical_coordinates_n[case1]) * dx) / x_velocity[case1]
        t1[case2] = -(x_particles[case2] - (logical_coordinates_n[case2] + 0.5) * dx) / x_velocity[case2]
        t1[case3] = ((logical_coordinates_n[case3] + 1) * dx - x_particles[case3]) / x_velocity[case3]
        t1[case4] = ((logical_coordinates_n[case4] + 0.5) * dx - x_particles[case4]) / x_velocity[case4]
        switches_cells = t1 < time

        logical_coordinates_depo = np.where(particle_in_right_half, logical_coordinates_n+1, logical_coordinates_n)
        time_in_this_iteration = np.where(switches_cells, t1, time)
        current_contribution = x_velocity * q / dt * time_in_this_iteration
        # if (~switches_cells).any():
        # current_contribution[~switches_cells] *= time[~switches_cells]
        # if switches_cells.any():
        # current_contribution[switches_cells] *= t1[switches_cells]

        new_value =np.bincount(logical_coordinates_depo + 1, current_contribution, minlength=j_x.size)
        j_x += new_value

        new_locations = np.empty_like(x_particles)
        new_locations[case1] = (logical_coordinates_n[case1]) * dx - epsilon
        new_locations[case2] = (logical_coordinates_n[case2] + 0.5) * dx - epsilon
        new_locations[case3] = (logical_coordinates_n[case3] + 1) * dx + epsilon
        new_locations[case4] = (logical_coordinates_n[case4] + 0.5) * dx + epsilon
        new_time = time - time_in_this_iteration

        #==============DEBUG===================
        # actives.append(active.sum())
        # print(f"{10*'='}ITERATION {counter}{10*'='}")
        # df = pandas.DataFrame()
        # # df['active'] = active
        # # df['in_left_half'] = particle_in_left_half
        # # df['in_right_half'] = particle_in_right_half
        # df['x_particles/dx'] = x_particles / dx
        # df['x_velocity/c'] = x_velocity / c
        # df['logical_pos'] = np.floor(x_particles / dx * 2) /2
        # df['logical_coordinates_n'] = logical_coordinates_n
        # df['logical_coordinates_depo'] = logical_coordinates_depo
        # df['current_contribution'] = current_contribution #/ x_velocity / q
        # df['current_contribution_normed'] = current_contribution / x_velocity / q
        # df['time/dt'] = time/dt
        # df['t1/dt'] = t1/dt
        # df['time_overflow/dt'] = new_time/dt
        # df['time_in_this/dt'] = time_in_this_iteration/dt
        # df['switches'] = switches_cells
        # case = np.zeros_like(x_particles, dtype=int)
        # case[case1] = 1
        # case[case2] = 2
        # case[case3] = 3
        # case[case4] = 4
        # print(case)
        # df['case'] = case
        # print(df)
        # # print(df[df['logical_coordinates_depo']==9])
        #
        # added_indices = new_value != 0
        # bins_df = pandas.DataFrame()
        # # bins_df['indices'] = np.where(logical_coordinates_depo +1)[added_indices]
        # bins_df['values'] = (new_value)[added_indices]
        # print(bins_df)
        #==============DEBUG===================
        active = switches_cells
        x_particles = new_locations[active]
        x_velocity = x_velocity[active]
        time = new_time[active]
        active = np.ones_like(x_particles, dtype=bool)

# @jit()
def aperiodic_longitudinal_current_deposition(j_x, x_velocity, x_particles, dx, dt, q):
    longitudinal_current_deposition(j_x, x_velocity, x_particles, dx, dt, q)
    j_x[0] = 0
    j_x[-2:] = 0

# @jit()
def periodic_longitudinal_current_deposition(j_x, x_velocity, x_particles, dx, dt, q):
    longitudinal_current_deposition(j_x, x_velocity, x_particles, dx, dt, q)
    j_x[-3] += j_x[0]
    j_x[0] = 0
    j_x[1:3] += j_x[-2:]
    j_x[-2:] = 0

# @jit()
def transversal_current_deposition(j_yz, velocity, x_particles, dx, dt, q):
    # print("Transversal deposition")
    epsilon = 1e-10 * dx
    time = np.ones_like(x_particles, dtype=np.float64) * dt
    counter = 0
    active = np.any(velocity[:,1:],axis=1)
    # dataframes = []
    while active.any():
        counter += 1
        if counter > 4:
            # import matplotlib.pyplot as plt
            # plt.plot(actives)
            # plt.show()
            raise Exception("Infinite recurrence!")
        # print(active.sum())
        x_velocity = velocity[:, 0]
        y_velocity = velocity[:, 1]
        z_velocity = velocity[:, 2]

        logical_coordinates_n = (x_particles // dx).astype(np.int32)
        particle_in_left_half = x_particles / dx - logical_coordinates_n < 0.5
        particle_in_right_half = ~particle_in_left_half

        velocity_to_left = x_velocity < 0
        velocity_to_right = x_velocity > 0
        velocity_zero = x_velocity == 0

        t1 = np.empty_like(x_particles)
        s = np.empty_like(x_particles)

        # REFACTOR: functionify upon reshuffling algorithm which I'm unsure I'm up for
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
        s[x_velocity == 0] = x_particles[x_velocity == 0]

        time_overflow = time - t1
        switches_cells = time_overflow > 0
        time_in_this_iteration = time.copy()
        time_in_this_iteration[switches_cells] = t1[switches_cells]
        time_in_this_iteration[x_velocity == 0] = dt

        jy_contribution = q * y_velocity / dt * time_in_this_iteration
        jz_contribution = q * z_velocity / dt * time_in_this_iteration

        # noinspection PyUnresolvedReferences
        sign = particle_in_left_half.astype(int) * 2 - 1
        distance_to_current_cell_center = (logical_coordinates_n + 0.5) * dx - x_particles
        s0 = (1 - sign * distance_to_current_cell_center / dx)
        change_in_coverage = sign * x_velocity * time_in_this_iteration / dx
        s1 = s0 + change_in_coverage
        w = 0.5 * (s0 + s1)
        # c1 = switches_cells & particle_in_left_half
        # c2 = ~switches_cells & particle_in_left_half
        # c3 = ~switches_cells & particle_in_right_half
        # c4 = switches_cells & particle_in_right_half
        # w = np.zeros_like(x_particles)
        # paren = (x_particles + 0.5 * x_velocity * time_in_this_iteration ) /dx
        # w[c1 | c2] = 0.5 - logical_coordinates_n[c1 | c2] + paren[c1 | c2]
        # w[c3 | c4] = 1.5 + logical_coordinates_n[c3 | c4] - paren[c3 | c4]

        logical_coordinates_depo = logical_coordinates_n.copy()
        logical_coordinates_depo[particle_in_left_half] -= 1
        logical_coordinates_depo[particle_in_right_half] += 1

        y_contribution_to_current_cell = w * jy_contribution
        z_contribution_to_current_cell = w * jz_contribution
        y_contribution_to_next_cell = (1 - w) * jy_contribution
        z_contribution_to_next_cell = (1 - w) * jz_contribution
        # y_contribution_to_next_cell = np.zeros_like(y_contribution_to_current_cell)
        # y_contribution_to_next_cell[c1 | c4] = (1 - w [c1|c4]) * q * y_velocity[c1 | c4] / dt * t1[c1 | c4]
        # y_contribution_to_next_cell[c2 | c3] = (1 - w [c2|c3]) * q * y_velocity[c2 | c3] / dt * time[c2 | c3]

        # assert (w <= 1).all(), f"w {w} > 1"
        # assert (w >= 0.5).all(), f"w {w} < 0.5"
        # assert (1-w <= 0.5).all(), f"1-w {1-w} > 0.5"
        # assert (1-w >= 0).all(), f"1-w {1-w} < 0"
        # if (~switches_cells).all():
        #     print(f"x0: {x_particles}\t v: {velocity}\t x1: {s} \t time: {time_in_this_iteration}")
        #     print("====J contribution=====")
        #     print(logical_coordinates_depo)
        #     print("y", y_contribution_to_current_cell)
        #     print("z", z_contribution_to_current_cell)
        #     print("to current cell:", logical_coordinates_n)
        #     print("y", y_contribution_to_next_cell)
        #     print("z", y_contribution_to_next_cell)
        #     print("to cell:", logical_coordinates_depo)

        j_yz[:, 0] += np.bincount(logical_coordinates_n + 2, y_contribution_to_current_cell, minlength=j_yz[:, 1].size)
        j_yz[:, 1] += np.bincount(logical_coordinates_n + 2, z_contribution_to_current_cell, minlength=j_yz[:, 1].size)

        j_yz[:, 0] += np.bincount(logical_coordinates_depo + 2, y_contribution_to_next_cell, minlength=j_yz[:, 1].size)
        j_yz[:, 1] += np.bincount(logical_coordinates_depo + 2, z_contribution_to_next_cell, minlength=j_yz[:, 1].size)

        # case = np.zeros_like(x_particles, dtype=int)
        # case[case1] = 1
        # case[case2] = 2
        # case[case3] = 3
        # case[case4] = 4
        # c= np.zeros_like(x_particles, dtype=int)
        # c[c1] = 1
        # c[c2] = 2
        # c[c3] = 3
        # c[c4] = 4
        # df = pandas.DataFrame(index=[counter])
        # df['active'] = active
        # df['case'] = case
        # df['c'] = c
        # df['x_particles/dx'] = x_particles/dx
        # # df['x_velocity'] = x_velocity
        # df['logical_pos'] = np.floor(x_particles / dx * 2) /2
        # df['time/dt'] = time/dt
        # df['t1/dt'] = t1/dt
        # df['toverflow/dt'] = time_overflow/dt
        # df['logical_coordinates_n'] = logical_coordinates_n
        # df['w'] = w
        # df['logical_coordinates_depo'] = logical_coordinates_depo
        # df['1-w'] = 1-w
        # df['time_in_this_iteration/dt'] = time_in_this_iteration/dt
        # df['s/dx'] = s/dx
        # # df['contribution_to_next'] = y_contribution_to_next_cell
        # # df['t1'] = t1
        # # df['time_overflow'] = time - t1
        # # df['velocity_zero'] = velocity_zero
        # df['switches'] = switches_cells
        # dataframes.append(df)
        # if counter > 2:
        #     string =f"""iteration:\t{counter}
        #           dx: {dx}
        #           number of actives: {active.sum()}
        #           fraction of case1 (in left, to left): {case1.sum() / active.sum()}
        #           fraction of case2 (in left, to right): {case2.sum() / active.sum()}
        #           fraction of case3 (in right, to right): {case3.sum() / active.sum()}
        #           fraction of case4 (in right, to left): {case4.sum() / active.sum()}
        #           x_particles: {x_particles},
        #           indices: {logical_coordinates_n},
        #           new locations: {s},
        #           t1 {t1},
        #           distance to cover in grid cell units: {x_velocity*time_overflow/dx},
        #           time left in dt units: {time_overflow/dt},
        #           distance to cover: {time_overflow*x_velocity},
        #           distance between current and next: {x_particles - s},
        #           \n\n"""
        #     modified_string = "\n".join(line.strip() for line in string.splitlines())
        #     print(modified_string)
        # print(counter,
        #       dx,
        #       active.sum(),
        #       case1.sum() / active.sum(),
        #       x_particles, # the first array in there
        #       logical_coordinates_n,
        #       s,
        #       # logical_coordinates_n - x_particles/dx,
        #       # (x_particles - logical_coordinates_n * dx)/dx,
        #       # (new_locations - logical_coordinates_n * dx)/dx,
        #       t1,
        #       x_velocity*time_overflow/dx,
        #       time_overflow/dt,
        #       time_overflow*x_velocity,
        #       x_particles - time_overflow,
        #       "\n\n"
        #       )
        active = switches_cells
        time = time_overflow[active]
        x_particles = s[active]
        velocity = velocity[active]
        active = np.ones_like(x_particles, dtype=bool)
    # print(pandas.concat(dataframes))
# @jit()
def aperiodic_transversal_current_deposition(j_yz, velocity, x_particles, dx, dt, q):
    transversal_current_deposition(j_yz, velocity, x_particles, dx, dt, q)
    j_yz[:2] = 0
    j_yz[-2:] = 0

# @jit()
def periodic_transversal_current_deposition(j_yz, velocity, x_particles, dx, dt, q):
    transversal_current_deposition(j_yz, velocity, x_particles, dx, dt, q)
    j_yz[-4:-2] += j_yz[:2]
    j_yz[2:4] += j_yz[-2:]
