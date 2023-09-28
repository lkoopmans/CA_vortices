import numpy as np
import random
import time
from pylab import *
import time
import copy
import imageio
import pickle
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc


# Define class which contains the core code of the model
class two_signaling_molecules:
    def __init__(self):
        self.relative_distance = []
        self.positions = []
        self.gridsize = 14
        self.N = 20 ** 2
        self.rcell = 0.2
        self.a0 = 1.5
        self.cell_topology = []
        self.cells = np.zeros((self.N, 2))
        self.periodic_bc = [0, 0]
        self.K = np.zeros((2, 2, 2))
        self.M = np.zeros((2, 2, 2))
        self.C = np.zeros((2, 1, 2))
        self.Coff = np.ones([2, 1, 2])
        self.C0n = []
        self.idx_celltype = []
        self.idx_nearest_neighbours = []
        self.lamb = []
        self.tmax = 1000
        self.cell_hist = np.zeros((20 ** 2, 2, 1000), np.int8)
        self.img = []
        self.Fn_a0 = 1
        self.double_topology_flag = False
        self.clock_start = time.time()
        self.clock_traject = np.zeros(10)
        self.number_of_nn = 6
        self.cell_4state = []
        self.cell_4phase = []
        self.state_fractions = []
        self.first_recurrent_state_time = []
        self.final_pattern_type = []
        self.vortex_cores = []
        self.Hxy = []
        self.Hxy_adjusted = []
        self.vortex_cores_labeled = []
        self.n_vortex = []
        self.xcom = []
        self.ycom = []
        self.cell_number_nn = []
        self.xmin_core = []
        self.ymin_core = []
        self.closest_pos = []
        self.charges = []
        self.idx_annihilation_moments = []
        self.idx_annihilation_moments_core_count = []

    def init_lattice(self, gridsize, periodic_bc, a0, rcell, type='triangular', path=''):
        n = gridsize ** 2
        self.a0 = a0
        self.gridsize = gridsize
        if type == 'triangular':

            [pos, lx, ly] = calculate_triangular_lattice(gridsize)
            dist = calculate_distance(pos, lx, ly, gridsize, periodic_bc)

            idx_nearest_neighbours = np.round(dist, 1) == 1

            row_coll_idx_nn = np.argwhere(idx_nearest_neighbours == 1)
            cell_number_nn = np.zeros((n, 7))

            cell_number_nn[:, 0] = np.arange(0, n)
            cell_number_nn[:, 1:] = row_coll_idx_nn[:, 1].reshape(n, 6)

            self.cell_number_nn = cell_number_nn.astype(int)
            self.idx_nearest_neighbours = idx_nearest_neighbours
            self.relative_distance = dist
            self.N = n
            self.positions = pos
            self.number_of_nn = 6

        elif type == 'excel':
            pos, dist, states = initialize_lattice_from_excel(path, periodic_bc)
            n = len(pos[:, 0])

            idx_nearest_neighbours = np.round(dist, 1) == 1

            row_coll_idx_nn = np.argwhere(idx_nearest_neighbours == True)

            if np.sum(row_coll_idx_nn == 1) == 4 * n:
                cell_number_nn = np.zeros((n, 5))
                cell_number_nn[:, 0] = np.arange(0, n)
                cell_number_nn[:, 1:] = row_coll_idx_nn[:, 1].reshape(n, 4)
            else:
                cell_number_nn = np.zeros((n, 5))
                cell_number_nn[:, 0] = np.arange(0, n)

            self.number_of_nn = 4
            self.cells = np.squeeze(get_2_number_seq(states[:, np.newaxis]))
            self.idx_nearest_neighbours = idx_nearest_neighbours
            self.cell_number_nn = cell_number_nn.astype(int)
            self.relative_distance = dist
            self.N = n
            self.positions = pos
        elif type == 'excel_triangular':
            pos_temp, dist_temp, states = initialize_lattice_from_excel(path, periodic_bc)
            n_cells = len(pos_temp[:, 0])

            gridsize = int(np.round(np.sqrt(n_cells)))

            [pos, lx, ly] = calculate_triangular_lattice(gridsize)
            dist = calculate_distance(pos, lx, ly, gridsize, periodic_bc)

            idx_nearest_neighbours = np.round(dist, 1) == 1

            row_coll_idx_nn = np.argwhere(idx_nearest_neighbours == True)

            if periodic_bc[0] == 1:
                cell_number_nn = np.zeros((n, 7))
                cell_number_nn[:, 0] = np.arange(0, n)
                cell_number_nn[:, 1:] = row_coll_idx_nn[:, 1].reshape(n, 6)
                self.cell_number_nn = cell_number_nn.astype(int)
            else:
                cell_number_nn = 0
                self.cell_number_nn = cell_number_nn

            self.number_of_nn = 6
            self.cells = np.squeeze(get_2_number_seq(states[:, np.newaxis]))
            self.idx_nearest_neighbours = idx_nearest_neighbours

            self.relative_distance = dist
            self.N = n
            self.positions = pos
        else:
            print('Invalid initialization type')

            self.gridsize = gridsize
            self.a0 = a0
            self.rcell = rcell
            self.periodic_bc = periodic_bc

        # In case the topology function is not called
        self.idx_celltype = np.zeros(n) == 0
        self.cell_topology = np.ones((gridsize, gridsize))

    def init_lattice_excel(self, path, a0, rcell):
        positions, r, states = initialize_lattice_from_excel(path, a0)
        n = len(positions[:, 0])

        self.cells = np.squeeze(get_2_number_seq(states[:, np.newaxis]))
        self.relative_distance = r
        self.N = n
        self.positions = positions
        self.gridsize = np.sqrt(n)
        self.a0 = a0
        self.rcell = rcell

        # In case the topology function is not called
        self.idx_celltype = np.zeros(n) == 0
        self.cell_topology = np.ones((self.gridsize, self.gridsize))

    def init_topology(self, topology_input):
        gridsize = self.gridsize
        temp = init_topology_mat(np.flip(topology_input), gridsize)
        top_mat = temp

        self.cell_topology = top_mat
        self.idx_celltype = top_mat.reshape(self.N, order='F') > 0

    def init_cell_state(self, p0, I_min, dI):
        dist = self.relative_distance
        N = self.N
        init_on = np.round(p0 * N)
        self.cells = init_I(init_on, self.a0, dist, N, I_min, dI)

    def init_general_parameters(self, K_matrix, C_matrix, M_matrix, lamb):
        self.K, self.double_topology_flag = force_input_matrix_shape(K_matrix, 1)
        M, self.double_topology_flag = force_input_matrix_shape(M_matrix, 1)
        self.Con, self.double_topology_flag = force_input_matrix_shape(C_matrix, 2)

        self.M = M.astype(int32)
        self.lamb = lamb

    def run_model(self, tmax):
        # extract parameters
        dist = self.relative_distance
        Rcell = self.rcell * self.a0
        a0 = self.a0
        lamb = self.lamb
        N = self.N
        idx_celltype = self.idx_celltype

        cells = self.cells
        Coff = self.Coff
        Con = self.Con

        M = self.M
        K = self.K

        self.cell_hist, self.tmax, self.Y = run_CA(M, K, N, tmax, dist, Rcell, a0, lamb, idx_celltype, Coff, Con, cells)

        # Clock moment 1
        self.clock_traject[1] = np.round(time.time() - self.clock_start, 2)

    def analyse_trajectory(self):
        # Define the two main representations: 4-state and spin-state
        self.cell_4state = get_4_number_seq(self.cell_hist)

        self.cell_4phase = get_phase_seq_4(self.cell_4state)

        # Compute the cell fractions
        self.state_fractions = compute_4_state_fractions(self)

        # Compute halting time: First time a configuration reoccurs
        self.first_recurrent_state_time = get_first_recurrent_state_time(self)

        # Compute final pattern type
        self.final_pattern_type = get_final_configuration_type(self)

        if self.number_of_nn == 6:
            # Clock moment 2
            self.clock_traject[2] = np.round(time.time() - self.clock_start, 2)

            # Compute all the spatial pi phase differences
            self.vortex_cores, self.Hxy, self.Hxy_adjusted = compute_phase_differences(self)

            # Clock moment 3
            self.clock_traject[3] = np.round(time.time() - self.clock_start, 2)

            # Labeling the vortex cores
            self.vortex_cores_labeled, self.n_vortex = label_triangular_lattice(self)

            # Clock moment 4
            self.clock_traject[4] = np.round(time.time() - self.clock_start, 2)

            # Compute centroids
            self.xcom, self.ycom = compute_centroid(self)

            # Clock moment 5
            self.clock_traject[5] = np.round(time.time() - self.clock_start, 2)

            # Post processing to correct for changing labels
            self.closest_pos, self.charges, self.xmin_core, self.ymin_core = compute_smallest_movement(self)

            # Clock moment 6
            self.clock_traject[6] = np.round(time.time() - self.clock_start, 2)

            # Compute other quantities
            self.idx_annihilation_moments, self.idx_annihilation_moments_core_count = compute_annihilation_moments(self)

    def clean_data(self):
        del self.C
        del self.C0n
        del self.Coff
        del self.Fn_a0
        del self.Y
        del self.double_topology_flag

    def show_trajectory(self, start_frame, frame_rate, store_frames=False, spin_vect=False):
        input_val_stored_frames = store_frames
        self.img = show_cells(self.tmax-start_frame, self.cell_hist[:, :, start_frame:], self.positions,
                              self.idx_celltype, frame_rate, spin_vect, self.vortex_cores_labeled, self.charges, self.cell_4phase, make_gif=True,
                              store_frames=input_val_stored_frames)

    def make_gif(self, frame_start, frame_end, frame_rate, file_name):
        img = self.img
        location = fr'C:\Users\larsk\Desktop\{file_name}.gif'

        imageio.mimsave(location, img[frame_start - 1:frame_end - 1], fps=frame_rate)


def compute_annihilation_moments(self):
    n_cores = self.n_vortex
    n_vortex = np.sum(self.charges != 0, axis=1)

    delta_n_vortex = np.diff(n_cores, axis=0) == -1
    delta_n_charge = np.diff(n_vortex, axis=0) == -2

    idx_annihilation_moments = np.argwhere(delta_n_vortex * delta_n_charge * (n_cores[:-1] == n_vortex[:-1]))
    idx_annihilation_moments_core_count = n_vortex[idx_annihilation_moments]

    return idx_annihilation_moments, idx_annihilation_moments_core_count


def compute_smallest_movement(self):
    t_end = self.tmax
    t_start = 0

    charges = np.zeros((t_end - t_start, 2 + int(np.max(self.vortex_cores_labeled))))
    CmX = np.zeros((t_end - t_start, 2 + int(np.max(self.vortex_cores_labeled))))
    CmY = np.zeros((t_end - t_start, 2 + int(np.max(self.vortex_cores_labeled))))

    if self.number_of_nn == 6:
        lx = np.max(self.positions[:, 0]) + self.positions[0, 0]/2
        ly = np.max(self.positions[:, 1]) + self.positions[0, 1]/2
    else:
        lx = np.max(self.positions[:, 0]) + self.positions[0, 0]
        ly = np.max(self.positions[:, 1]) + self.positions[0, 1]

    closest_pos = np.zeros((self.tmax, 20)) * np.nan
    for p in range(t_end - t_start):
        # Set frame number
        fn = p + t_start

        # Set frame and contours of frame
        hist_run = self.cell_4phase[:, fn]
        c_run = self.vortex_cores_labeled[:, fn]

        # Run through all labels
        idx_all = np.unique(self.vortex_cores_labeled[:, fn], axis=0)

        for idx_label in idx_all[1:]:
            x_hat, y_hat, c_r, m_x_r, m_y_r = get_contour_pos_relative_2_center(c_run, hist_run, idx_label, self)

            theta_r = np.arctan2(y_hat, x_hat) + np.pi
            idx_sort = np.argsort(theta_r)

            cwise = c_r[idx_sort]
            cwise_wrap = np.zeros((1, len(cwise[0, :]) + 1))
            cwise_wrap[0, :-1] = cwise
            cwise_wrap[0, len(cwise[0, :])] = cwise[0, 0]

            temp = np.diff(cwise_wrap)
            temp[temp / np.pi == -3 / 2] = 0.5 * np.pi
            temp[temp / np.pi == 3 / 2] = -0.5 * np.pi

            charges[p, idx_label - 1] = np.sum(temp) / np.pi
            CmX[p, idx_label - 1] = m_x_r
            CmY[p, idx_label - 1] = m_y_r

        # if p > 0:
        #     tempX = CmX[p, :]
        #     tempY = CmY[p, :]
        #     tempX_min = CmX[p - 1, :]
        #     tempY_min = CmY[p - 1, :]
        #
        #     idx_true = charges[p-1, :] != 0
        #
        #     dx = (tempX[idx_true, np.newaxis].T - tempX_min[idx_true, np.newaxis])
        #     dy = (tempY[idx_true, np.newaxis].T - tempY_min[idx_true, np.newaxis])
        #
        #     if self.periodic_bc[0] == 1:
        #         dx = (dx + lx / 2) % lx - lx / 2
        #
        #     if self.periodic_bc[1] == 1:
        #         dy = (dy + ly / 2) % ly - ly / 2
        #
        #     dist = (dx ** 2 + dy ** 2) ** 0.5
        #
        #     if dist.size != 0:
        #         closest_pos[p, :len(dist[0, :])] = np.argmin(dist, axis=0)

    return closest_pos, charges, CmX, CmY


def interchange_numbers(input_array, v1, v2):
    idx_v1 = input_array == v1
    idx_v2 = input_array == v2

    input_array[idx_v1] = v2
    input_array[idx_v2] = v1

    return input_array


def compute_phase_differences(self):
    # using integer numbers to speed up computations
    C4 = self.cell_4state.astype(np.int8)
    C4 = interchange_numbers(C4, 4, 3)

    temp = C4[self.cell_number_nn.ravel(), :]

    frame_values = temp.reshape(self.N, self.number_of_nn + 1, self.tmax)

    temp = frame_values[:, 0, :]
    delta = frame_values[:, 1:] - temp[:, np.newaxis, :]

    # Determine cos of the phase differences
    cos_nn = lut_Hxy(delta)
    xy_adjusted_nn = lut_Hxy_adjusted(delta)

    # Assign integer value to vortex cores
    vortex_cores = (np.squeeze(np.sum(abs(delta) == 2, axis=1)) > 0) * 1

    # Compute Hamiltonians from data
    Hxy = -np.squeeze(np.sum(cos_nn, axis=1)) * 1 / (self.N * self.number_of_nn)
    Hxy_adjusted = -1 / 2 - 2 * np.squeeze(np.sum(xy_adjusted_nn, axis=1)) * 1 / (self.N * self.number_of_nn)

    return vortex_cores, np.sum(Hxy, axis=0), np.sum(Hxy_adjusted, axis=0)


def lut_Hxy(values):
    # mimics the cosine for {-pi*3/2, -pi, -pi/2, 0, pi/2, pi, pi*3/2}

    values = abs(values)
    idx0 = values == 0
    idx1 = values == 1
    idx2 = values == 2
    idx3 = values == 3

    values[idx0] = 1
    values[idx1] = 0
    values[idx2] = -1
    values[idx3] = 0

    return values


def lut_Hxy_adjusted(values):
    # mimics the Xy Hamiltonain with corrected term for {-pi*3/2, -pi, -pi/2, 0, pi/2, pi, pi*3/2}

    values = abs(values)
    idx0 = values == 0
    idx1 = values == 1
    idx2 = values == 2
    idx3 = values == 3

    values[idx0] = 0
    values[idx1] = 0
    values[idx2] = -1
    values[idx3] = 0

    return values


def get_final_configuration_type(self):
    if len(np.unique(self.cell_4state[:, -1])) == 1:
        final_pattern_type = 1
    elif len(np.unique(np.diff(
            self.state_fractions[1, self.first_recurrent_state_time:]))) == 1:  # constant fractions after halting
        final_pattern_type = 3
    else:
        final_pattern_type = 2

    return final_pattern_type


def get_first_recurrent_state_time(self):
    cell4 = self.cell_4state

    unique_configurations = np.unique(cell4, axis=1)
    sz = unique_configurations.shape
    p_time = sz[1]

    return p_time


def compute_4_state_fractions(self):
    state_fractions = np.zeros((4, self.tmax))
    state_fractions[0, :] = np.sum(self.cell_4state == 1, axis=0) / self.N
    state_fractions[1, :] = np.sum(self.cell_4state == 2, axis=0) / self.N
    state_fractions[2, :] = np.sum(self.cell_4state == 3, axis=0) / self.N
    state_fractions[3, :] = np.sum(self.cell_4state == 4, axis=0) / self.N

    return state_fractions


# Running the cellular automaton
def run_CA(M, K, N, tmax, dist, Rcell, a0, lamb, idx_celltype, Coff, Con, cells):
    M_int = np.transpose(M)
    K = np.transpose(K)

    t = 0

    cells_hist = np.zeros((N, 2, tmax), np.int8)
    idx = dist > 0

    M = np.ones((N, N, 2))
    Y = np.ones((N, 2, tmax))

    for k in range(2):
        M[idx, k] = np.sinh(Rcell) / (a0 * dist[idx] / lamb[k]) * np.exp((Rcell - a0 * dist[idx]) / lamb[k])

    period = False

    while t < tmax and not period:
        cells_hist[:, :, t] = cells
        idx_loop = idx_celltype

        out = np.zeros((N, 4))
        C0 = np.zeros((N, 2))
        cells_out = np.zeros((N, 2))

        C0[idx_loop, :] = Coff[0, :, :] + (Con[0, :, :] - Coff[0, :, :]) * cells[idx_loop, :]
        C0[np.invert(idx_loop), :] = Coff[1, :, :] + (Con[1, :, :] - Coff[1, :, :]) * cells[np.invert(idx_loop), :]

        for k in range(2):
            Y[:, k, t] = np.matmul(np.squeeze(M[:, :, k]), C0[:, k])

        for j in range(2):
            out[idx_loop, 0] = ((Y[idx_loop, 0, t] - np.squeeze(K[0, 0, j])) * np.squeeze(M_int[0, 0, j]) > 0) + (
                    1 - abs(M_int[0, 0, j]))
            out[idx_loop, 1] = ((Y[idx_loop, 1, t] - np.squeeze(K[1, 0, j])) * np.squeeze(M_int[1, 0, j]) > 0) + (
                    1 - abs(M_int[1, 0, j]))
            out[idx_loop, 2] = ((Y[idx_loop, 0, t] - np.squeeze(K[0, 1, j])) * np.squeeze(M_int[0, 1, j]) > 0) + (
                    1 - abs(M_int[0, 1, j]))
            out[idx_loop, 3] = ((Y[idx_loop, 1, t] - np.squeeze(K[1, 1, j])) * np.squeeze(M_int[1, 1, j]) > 0) + (
                    1 - abs(M_int[1, 1, j]))
            cells_out[idx_loop, 0] = np.squeeze(out[idx_loop, 0]) * np.squeeze(out[idx_loop, 1])
            cells_out[idx_loop, 1] = np.squeeze(out[idx_loop, 2]) * np.squeeze(out[idx_loop, 3])

            idx_loop = np.invert(idx_loop)

        t += 1
        cells = cells_out

    return cells_hist, t, Y


def label_triangular_lattice(self):
    tmax = self.tmax
    vortex_cores = copy.deepcopy(self.vortex_cores)

    # Apply two-pass algorithm writen for a triangular lattice
    for frame_number in range(0, tmax):
        # phase 1: first labeling
        labels = np.where(vortex_cores[:, frame_number] == 1)
        for i in range(len(labels[0][:])):
            current_cell = vortex_cores[labels[0][i], frame_number]

            if current_cell == 1:
                neighbours_idx = np.where(np.round(self.relative_distance[labels[0][i], :], 1) == 1)
                neighbours_values = vortex_cores[neighbours_idx, frame_number]
                temp = neighbours_values[neighbours_values > 1]

                if not any(temp):
                    min_neighbour_value = 1
                else:
                    min_neighbour_value = min(temp)

                if min_neighbour_value == 1:
                    vortex_cores[labels[0][i], frame_number] = np.max(vortex_cores[:, frame_number]) + 1
                else:
                    vortex_cores[labels[0][i], frame_number] = min_neighbour_value

        # phase 2: assign minimum to all connected and labeled areas
        for i in range(len(labels[0][:])):
            neighbours_idx = np.where(np.round(self.relative_distance[labels[0][i], :], 1) <= 1)
            neighbours_values = vortex_cores[neighbours_idx, frame_number]
            neighbours_unique = np.unique(neighbours_values[neighbours_values > 0])
            if len(neighbours_unique) > 1:
                for j in range(len(neighbours_unique)):
                    vortex_cores[vortex_cores[:, frame_number] == neighbours_unique[j], frame_number] = \
                        np.min(neighbours_unique)

        # phase 3: Make sure no index is missed (e.g. [1 2 4 5] --> [1 2 3 4])
        # more elegant: np.unique(a, return_inverse=True)[1].reshape(a.shape)

        for i in range(np.max(vortex_cores[:, frame_number])):
            if not any(vortex_cores[:, frame_number] == (i + 0)):
                vortex_cores[vortex_cores[:, frame_number] >= (i + 0), frame_number] -= 1

    vortex_cores_labeled = vortex_cores
    n_vortex = np.max(vortex_cores, axis=0)

    return vortex_cores_labeled, n_vortex


def label_elements_triangular_lattice(input_images, relative_distance, t_start, t_end):

    # Apply two-pass algorithm writen for a triangular lattice
    for frame_number in range(t_start, t_end):
        # phase 1: first labeling
        labels = np.where(input_images[:, frame_number] == 1)
        for i in range(len(labels[0][:])):
            current_cell = input_images[labels[0][i], frame_number]

            if current_cell == 1:
                neighbours_idx = np.where(np.round(relative_distance[labels[0][i], :], 1) == 1)
                neighbours_values = input_images[neighbours_idx, frame_number]
                temp = neighbours_values[neighbours_values > 1]

                if not any(temp):
                    min_neighbour_value = 1
                else:
                    min_neighbour_value = min(temp)

                if min_neighbour_value == 1:
                    input_images[labels[0][i], frame_number] = np.max(input_images[:, frame_number]) + 1
                else:
                    input_images[labels[0][i], frame_number] = min_neighbour_value

        # phase 2: assign minimum to all connected and labeled areas
        for i in range(len(labels[0][:])):
            neighbours_idx = np.where(np.round(relative_distance[labels[0][i], :], 1) <= 1)
            neighbours_values = input_images[neighbours_idx, frame_number]
            neighbours_unique = np.unique(neighbours_values[neighbours_values > 0])
            if len(neighbours_unique) > 1:
                for j in range(len(neighbours_unique)):
                    input_images[input_images[:, frame_number] == neighbours_unique[j], frame_number] = \
                        np.min(neighbours_unique)

        # phase 3: Make sure no index is missed (e.g. [1 2 4 5] --> [1 2 3 4])
        # more elegant: np.unique(a, return_inverse=True)[1].reshape(a.shape)

        for i in range(np.max(input_images[:, frame_number])):
            if not any(input_images[:, frame_number] == (i + 0)):
                input_images[input_images[:, frame_number] >= (i + 0), frame_number] -= 1

    images_labeled = input_images[:, t_start:t_end]
    number_of_elements = np.max(input_images, axis=0)

    return images_labeled, number_of_elements


def compute_wrap_coefficients(trajectory, fn_start, fn_end):

    periodic_neighbours = np.round(trajectory.relative_distance, 1) <= 1

    # compute relative distance again but now with non-periodic bc
    [x1, x2] = np.meshgrid(trajectory.positions[:, 0], trajectory.positions[:, 0])
    [y1, y2] = np.meshgrid(trajectory.positions[:, 1], trajectory.positions[:, 1])

    dx = x1 - x2
    dy = y1 - y2

    r = (dx ** 2 + dy ** 2) ** 0.5
    r = r / r[0, 1]

    non_periodic_neighbours = np.round(r, 1) <= 1
    trajectory.relative_distance = r

    edge_cells = ~non_periodic_neighbours * periodic_neighbours

    y_top = trajectory.positions[:, 1] == np.max(trajectory.positions[:, 1])
    x_top = np.logical_or(trajectory.positions[:, 0] == trajectory.positions[-1, 0],
                          trajectory.positions[:, 0] == trajectory.positions[-2, 0])

    x_top_pos = np.squeeze(np.array(np.where(x_top)))
    y_top_pos = np.squeeze(np.array(np.where(y_top)))

    wrap_all = np.zeros((fn_end-fn_start, 2))

    for j in range(fn_end-fn_start):
        fn = fn_start + j

        frame = trajectory.cell_4state[:, fn]
        frame_not_4_state = (frame != 4) * 1

        trajectory.vortex_cores = frame_not_4_state[:, np.newaxis]

        trajectory.tmax = 1
        label_values, nn = label_triangular_lattice(trajectory)

        x_wrap = 0
        c = 0

        while x_wrap == 0 | c < trajectory.gridsize:
            x_wrap = any(label_values[x_top_pos[c]] == label_values[edge_cells[x_top_pos[c], :]])
            c += 1

        wrap_all[j, 0] = x_wrap

        y_wrap = 0
        c = 0

        while y_wrap == 0 | c < trajectory.gridsize:
            y_wrap = any(label_values[y_top_pos[c]] == label_values[edge_cells[y_top_pos[c], :]])
            c += 1

        wrap_all[j, 1] = y_wrap

    return wrap_all


def calculate_triangular_lattice(gridsize):
    lx = 1

    delx = lx / gridsize
    dely = np.sqrt(3) / 2 * delx
    ly = dely * gridsize

    x = np.linspace(0, gridsize - 1, gridsize)
    xm, ym = np.meshgrid(x, x)

    x = (xm + np.mod(ym, 2) / 2) * delx
    y = ym * dely

    pos = np.column_stack((x.flatten('F'), y.flatten('F')))

    return pos, lx, ly


def calculate_rectangular_lattice(gridsize):
    lx = 1

    delx = lx / gridsize
    dely = np.sqrt(3) / 2 * delx
    ly = dely * gridsize

    x = np.linspace(0, gridsize - 1, gridsize)
    xm, ym = np.meshgrid(x, x)

    x = (xm + np.mod(ym, 2) / 2) * delx
    y = ym * dely

    pos = np.column_stack((x.flatten('F'), y.flatten('F')))

    return pos, lx, ly


def calculate_distance(pos, lx, ly, gz, periodic_bc):
    [x1, x2] = np.meshgrid(pos[:, 0], pos[:, 0])
    [y1, y2] = np.meshgrid(pos[:, 1], pos[:, 1])

    dx = np.mod(abs(x1 - x2), lx)
    dy = np.mod(abs(y1 - y2), ly)

    if periodic_bc[0] == 1:
        dx[dx > (lx - dx)] = lx - dx[dx > (lx - dx)]

    if periodic_bc[1] == 1:
        dy[dy > (ly - dy)] = ly - dy[dy > (ly - dy)]

    dist = (dx ** 2 + dy ** 2) ** 0.5

    dist = dist / (lx / gz)

    return dist


def init_topology_mat(topology_mat_in, gz):
    Sx = len(topology_mat_in)
    Sx_scale = np.floor(gz / Sx)
    Sx_res = int(gz - Sx * Sx_scale)
    Sx_vector = [None] * Sx

    Sy = len(topology_mat_in[0])
    Sy_scale = np.floor(gz / Sy)
    Sy_res = int(gz - Sy * Sy_scale)
    Sy_vector = [None] * Sy

    for i in range(Sx):
        if Sx_res != 0:
            Sx_vector[i] = (Sx_scale + 1)
            Sx_res -= 1
        else:
            Sx_vector[i] = Sx_scale

    for i in range(Sy):
        if Sy_res != 0:
            Sy_vector[i] = Sy_scale + 1
            Sy_res -= 1
        else:
            Sy_vector[i] = Sy_scale

    temp = np.repeat(topology_mat_in, Sx_vector, axis=0)
    Topology_mat = np.repeat(temp, Sy_vector, axis=1)

    return Topology_mat


def init_I(init_on, a0, dist, N, I_min, dI):
    cells = np.zeros((N, 2))
    for idx in range(2):
        cells[:int(init_on[idx]), idx] = 1
        cells[:, idx] = np.random.permutation(cells[:, idx])

        maxsteps = 5000
        I, theta = calculate_moranI(cells[:, idx], a0 * dist)

        if sum(cells[:, idx]) == N or sum(cells[:, idx]) == 0 or sum(cells[:, idx]) == 1 or sum(cells[:, idx]) == N - 1:
            check = False
        else:
            check = True

        eps = 1e-5
        dist_vec = calculate_unique_distances(dist, eps)
        dist1 = dist_vec[1]
        temp1 = dist1 + eps > dist
        temp2 = dist > dist1 - eps
        first_nei_idx = temp1 * temp2

        first_nei = np.zeros_like(first_nei_idx)
        first_nei[first_nei_idx] = 1
        increase = (I < I_min)
        t = 0

        I_max = I_min + dI
        while (I < I_min or I > I_max) and t < maxsteps and check:
            t = t + 1
            cells_new = cells[:, idx]

            nei_ON = np.matmul(first_nei, cells_new)
            nei_ON_1 = nei_ON[cells_new > 0]
            cond1 = 1

            if increase:
                idx_temp = res = np.argwhere(nei_ON_1 < 3)
                if idx_temp.size == 0:
                    idx_temp = np.argwhere(nei_ON_1 == np.min(nei_ON_1))
                    cond1 = 0
            else:
                idx_temp = res = np.argwhere(nei_ON_1 > 3)
                if idx_temp.size == 0:
                    idx_temp = np.argwhere(nei_ON_1 == np.max(nei_ON_1))
                    cond1 = 0.5

            idx_ON = np.random.choice(idx_temp.ravel())
            allON = np.argwhere(cells_new == 1)
            idx_1 = allON[:int(idx_ON)]

            nei_ON_0 = nei_ON[cells_new < 1]
            cond2 = 1

            if increase:
                idx_temp = res = np.argwhere(nei_ON_0 > 3)
                if idx_temp.size == 0:
                    idx_temp = np.argwhere(nei_ON_0 == np.min(nei_ON_0))
                    cond2 = 0
            else:
                idx_temp = res = np.argwhere(nei_ON_0 < 3)
                if idx_temp.size == 0:
                    idx_temp = np.argwhere(nei_ON_0 == np.max(nei_ON_0))
                    cond2 = 0

            idx_OFF = np.random.choice(idx_temp.ravel())
            allOFF = np.argwhere(cells_new == 0)
            idx_0 = allOFF[:int(idx_OFF)]

            if increase:
                idx_inc = 1
                cells_new[idx_0[-idx_inc:]] = abs(cells_new[idx_0[-idx_inc:]] - 1)
                cells_new[idx_1[-idx_inc:]] = abs(cells_new[idx_1[-idx_inc:]] - 1)
            else:
                cells_new[idx_0[-1:]] = abs(cells_new[idx_0[-1:]] - 1)
                cells_new[idx_1[-1:]] = abs(cells_new[idx_1[-1:]] - 1)

            [I_new, theta] = calculate_moranI(cells_new, a0 * dist)

            if cond1 == 1 and cond2 == 1:
                cells[:, idx] = cells_new
                I = I_new
            elif increase and (I_new >= I):
                cells[:, idx] = cells_new
                I = I_new
            elif not increase and I_new <= I:
                cells[:, idx] = cells_new
                I = I_new

            increase = (I < I_min)

    return cells


def calculate_moranI(cells, dist):
    cells_pm = 2 * cells - 1
    cell_mean = np.mean(cells_pm)

    [cells_matx, cells_maty] = np.meshgrid(cells_pm, cells_pm)

    idx = dist > 0

    M = np.zeros_like(dist)
    M[idx] = (np.exp(-dist[idx]) / dist[idx])

    w_summed = np.sum(np.exp(-dist[idx]) / dist[idx])

    theta = np.sum(M * cells_matx * cells_maty) / w_summed

    temp = np.sum(M * (cells_matx - cell_mean) * (cells_maty - cell_mean))

    if temp != 0:
        cells_var = np.var(cells_pm, axis=0)
        I = np.sum(temp) / w_summed / cells_var
    else:
        I = 0

    return I, theta


def calculate_unique_distances(dist, eps):
    round_value = int(abs(np.log(eps) / np.log(10)))
    distance_value = np.unique(np.round(dist, round_value))

    return distance_value


def force_input_matrix_shape(M_in, type):
    sz = M_in.shape
    double_topology_flag = False

    if type == 1:
        M_out = np.zeros((2, 2, 2))

        if len(sz) == 3 and sz[2] == 1:
            M_out[0, :, :] = M_in
            M_out[1, :, :] = M_in
        elif len(sz) == 2:
            M_out[0, :, :] = M_in
            M_out[1, :, :] = M_in
        elif len(sz) == 3 and sz[2] == 2:
            M_out = M_in
            double_topology_flag = True
        else:
            print('Incorrect input size')
    elif type == 2:
        M_out = np.zeros((2, 1, 2))
        if len(sz) == 3 and sz[2] == 1:
            M_out[0, :, :] = M_in.T
            M_out[1, :, :] = M_in.T
        elif len(sz) == 1:
            M_out[0, :, :] = M_in
            M_out[1, :, :] = M_in
        elif len(sz) == 3 and sz[2] == 2:
            M_out = M_in
        else:
            print('Incorrect input size')

    else:
        M_out = []
        print('Incorrect input size')
    return M_out, double_topology_flag


def show_cells_single_frame(frame, trajectory, gain=1):
    # init background and set layout plot
    bg_brightness = 240

    plt.ion()
    fig, ax = plt.subplots()

    ax.set_facecolor((bg_brightness / 255, bg_brightness / 255, bg_brightness / 255))
    fig.tight_layout()
    plt.xticks([])
    plt.yticks([])

    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    delta = 0.1

    cells = trajectory.cell_hist[:, :, frame]

    gain *= (6000 / np.sqrt(len(cells)))

    colours1 = np.asarray([None] * len(trajectory.positions[:, 0]), dtype=object)

    colours1[:] = 'blue'

    pos1 = trajectory.positions

    plt_state1_1 = plt.scatter(pos1[:, 0], pos1[:, 1], s=gain, c='blue', marker="o", edgecolors='black')

    plt.xlim(0 - delta, np.max(pos1[:, 0]) + delta)
    plt.ylim(0 - delta, np.max(pos1[:, 1]) + delta)

    colours1[(cells[:, 0] == 0) * (cells[:, 1] == 0)] = 'blue'
    colours1[(cells[:, 0] == 0) * (cells[:, 1] == 1)] = 'red'
    colours1[(cells[:, 0] == 1) * (cells[:, 1] == 0)] = 'white'
    colours1[(cells[:, 0] == 1) * (cells[:, 1] == 1)] = 'black'

    plt_state1_1.set_facecolors(c=colours1.tolist())

    plt.title('Time step=%i' % (frame))

    fig.canvas.draw()


def show_cells(tmax, cells_hist, pos, idx_celltype, frame_rate, spin_vect, cores_labeled, charges, cell_4phase, make_gif=False, store_frames=False):

    make_gif = True
    # spin_vect = True
    img = [None] * tmax

    g_vector = 2

    # init background and set layout plot
    bg_brightness = 240

    plt.ion()
    fig, ax = plt.subplots()

    ax.set_facecolor((bg_brightness / 255, bg_brightness / 255, bg_brightness / 255))
    fig.tight_layout()
    plt.xticks([])
    plt.yticks([])

    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    delta = 0.6

    cells = cells_hist[:, :, 0]

    gain = 6000/np.sqrt(len(cells_hist[:, 0]))
    idx_celltype1 = idx_celltype
    idx_celltype2 = np.invert(idx_celltype)

    colours1 = np.asarray([None] * len(pos[idx_celltype1, 0]), dtype=object)
    colours2 = np.asarray([None] * len(pos[idx_celltype2, 0]), dtype=object)

    colours1[:] = 'blue'
    colours2[:] = 'blue'

    pos1 = pos[idx_celltype1, :]
    pos2 = pos[idx_celltype2, :]

    if not spin_vect:
        plt_state1_1 = plt.scatter(pos1[:, 0], pos1[:, 1], s=gain, c='blue', marker="o", edgecolors='black')
        plt_state1_2 = plt.scatter(pos2[:, 0], pos2[:, 1], s=gain, c='blue', marker="s", edgecolors='black')
        plt.show()

        colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 0)] = 'blue'
        colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 1)] = 'red'
        colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 0)] = 'white'
        colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 1)] = 'black'

        colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 0)] = 'blue'
        colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 1)] = 'red'
        colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 0)] = 'white'
        colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 1)] = 'black'

        plt_state1_1.set_facecolors(c=colours1.tolist())
        plt_state1_2.set_facecolors(c=colours2.tolist())
    else:
        plt.quiver(pos[:, 0], pos[:, 1], np.cos(cell_4phase[:, 0]), np.sin(cell_4phase[:, 0]))
        plt.scatter(pos[:, 0], pos[:, 1], c=cores_labeled[:, 0], edgecolors='black', cmap=plt.get_cmap('hot'), s=80)
        plt.show()

    plt.xlim(0 - delta*2, np.max(pos[:, 0]) + delta*2)
    plt.ylim(0 - delta, np.max(pos[:, 1]) + delta)

    fig.canvas.draw()
    plt.show()

    if make_gif:
        temp = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        img[0] = temp.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    for t in range(1, tmax):
        t1 = time.time()
        print(t)
        plt.title('Time step=%i' % (t + 1 ))

        if not(spin_vect):
            cells = cells_hist[:, :, t]

            colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 0)] = 'blue'
            colours1[(cells[idx_celltype1, 0] == 0) * (cells[idx_celltype1, 1] == 1)] = 'red'
            colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 0)] = 'white'
            colours1[(cells[idx_celltype1, 0] == 1) * (cells[idx_celltype1, 1] == 1)] = 'black'

            plt_state1_1.set_facecolors(c=colours1.tolist())

            colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 0)] = 'blue'
            colours2[(cells[idx_celltype2, 0] == 0) * (cells[idx_celltype2, 1] == 1)] = 'red'
            colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 0)] = 'white'
            colours2[(cells[idx_celltype2, 0] == 1) * (cells[idx_celltype2, 1] == 1)] = 'black'

            plt_state1_2.set_facecolors(c=colours2.tolist())
            fig.canvas.draw()
            plt.show()
        else:
            plt.clf()
            plt.quiver(pos[:, 0], pos[:, 1], np.cos(cell_4phase[:, t]), np.sin(cell_4phase[:, t]))
            plt.scatter(pos[:, 0], pos[:, 1], c='white', edgecolors='black',
                        cmap=plt.get_cmap('hot'), s=80)
            for p in unique(cores_labeled[:, t]):
                if p > 0:
                    temp_label = cores_labeled[:, t] == p
                    if charges[t, p-1] < 0:
                        plt.scatter(pos[temp_label, 0], pos[temp_label, 1], c='orange', edgecolors='black', cmap=plt.get_cmap('hot'), s=80)
                    elif charges[t, p-1] > 0:
                        plt.scatter(pos[temp_label, 0], pos[temp_label, 1], c='blue', edgecolors='black',
                                    cmap=plt.get_cmap('hot'), s=80)
                    else:
                        plt.scatter(pos[temp_label, 0], pos[temp_label, 1], c='purple', edgecolors='black',
                                    cmap=plt.get_cmap('hot'), s=80)
                    plt.title('T=' + str(t))
        fig.canvas.flush_events()
        plt.show()
        t2 = time.time()
        delta_time = t2 - t1

        if make_gif:
            temp = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            img[t] = temp.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if store_frames:
            plt.savefig('Frames_saved_eps/step'+str(t)+'.eps', format='eps')

        if delta_time < 1 / frame_rate:
            time.sleep(1 / frame_rate - delta_time)

    return img

def copy_CA(CA_in):
    CA_out = copy.deepcopy(CA_in)
    return CA_out


def add_defects(CA_in, mode, *argv):
    CA_defect = copy.deepcopy(CA_in)

    if mode == 'random':
        randy = int(np.round(random.uniform(0, len(CA_defect.cell_hist[:, 0]) - 1)))
        randx = int(np.round(random.uniform(0, 1)))
        CA_defect.cell_hist[randy, randx] = abs(CA_defect.cell_hist[randy, randx] - 1)
    elif mode == 'fixed':
        idx_fixed = argv
        CA_defect.cell_hist[idx_fixed[0], idx_fixed[1]] = abs(CA_defect.cell_hist[idx_fixed[0], idx_fixed[1]] - 1)
    else:
        print('No changes have been made, select a correct mode type')

    return CA_defect


def save_trajectory(file, file_path, file_name):
    path = fr'\\NAS\homes\Lars\MEP_Data\{file_path}\{file_name}.pkl'

    with open(path, 'wb') as output:
        pickle.dump(file, output, pickle.HIGHEST_PROTOCOL)


def load_trajectory(path):
    # path = fr'C:\Users\larsk\Documents\1. TU delft\1. Vakken\MEP\Python\MEP\{file_path}\{file_name}.pkl'

    with open(path, 'rb') as input:
        output = pickle.load(input)

    return output


def get_4_number_seq(cell_hist):
    cell_hist_out = np.zeros((len(cell_hist[:, 0, 0]), len(cell_hist[0, 0, :])))

    cell_hist_out[(cell_hist[:, 0, :] == 0) * (cell_hist[:, 1, :] == 0)] = 1
    cell_hist_out[(cell_hist[:, 0, :] == 0) * (cell_hist[:, 1, :] == 1)] = 2
    cell_hist_out[(cell_hist[:, 0, :] == 1) * (cell_hist[:, 1, :] == 0)] = 3
    cell_hist_out[(cell_hist[:, 0, :] == 1) * (cell_hist[:, 1, :] == 1)] = 4

    return cell_hist_out


def get_phase_seq_4(cell_hist):
    cell_hist_out = np.zeros((len(cell_hist[:, 1]), len(cell_hist[1, :])))

    cell_hist_out[cell_hist == 1] = 0
    cell_hist_out[cell_hist == 2] = 1 / 2 * np.pi
    cell_hist_out[cell_hist == 4] = 1 * np.pi
    cell_hist_out[cell_hist == 3] = 3 / 2 * np.pi

    return cell_hist_out


def get_2_number_seq(cell_hist):
    cell_hist_out = np.zeros((len(cell_hist[:, 0]), len(cell_hist[0, :]), 2))

    cell_hist_out[cell_hist == 1, 0] = 0
    cell_hist_out[cell_hist == 1, 1] = 0

    cell_hist_out[cell_hist == 2, 0] = 0
    cell_hist_out[cell_hist == 2, 1] = 1

    cell_hist_out[cell_hist == 3, 0] = 1
    cell_hist_out[cell_hist == 3, 1] = 0

    cell_hist_out[cell_hist == 4, 0] = 1
    cell_hist_out[cell_hist == 4, 1] = 1

    return cell_hist_out


def get_probability_matrix(CA1, start, stop):
    histo = np.zeros((len(CA1.cell_hist[:, 1, 1]), 1 + len(CA1.cell_hist[1, 1, start:stop])))
    histo[:, :-1] = get_4_number_seq(CA1.cell_hist[:, :, start:stop])

    idx1 = np.where(histo == 1)
    idx2 = np.where(histo == 2)
    idx3 = np.where(histo == 3)
    idx4 = np.where(histo == 4)

    idx1_valnext = [idx1[0], idx1[1] + 1]
    idx2_valnext = [idx2[0], idx2[1] + 1]
    idx3_valnext = [idx3[0], idx3[1] + 1]
    idx4_valnext = [idx4[0], idx4[1] + 1]

    valnext1 = histo[idx1_valnext[0], idx1_valnext[1]]
    valnext2 = histo[idx2_valnext[0], idx2_valnext[1]]
    valnext3 = histo[idx3_valnext[0], idx3_valnext[1]]
    valnext4 = histo[idx4_valnext[0], idx4_valnext[1]]

    bins = np.zeros((4, 4))

    for c in range(4):
        bins[0, c] = np.sum(valnext1 == c + 1)
        bins[1, c] = np.sum(valnext2 == c + 1)
        bins[2, c] = np.sum(valnext3 == c + 1)
        bins[3, c] = np.sum(valnext4 == c + 1)

    norm = np.sum(bins, axis=1)
    idx_nonz = norm != 0
    bins_norm = np.zeros((4, 4))
    bins_norm[idx_nonz] = np.round(bins[idx_nonz] * 1 / np.tile(norm[idx_nonz], [4, 1]).transpose(), 7)

    return bins_norm


def get_cell_neighbours(CA1, cell_number):
    neighbours_idx = np.where(np.round(CA1.relative_distance[cell_number, :], 1) == 1)
    cell4 = get_4_number_seq(CA1.cell_hist)
    neighbours = cell4[neighbours_idx, :]

    return neighbours


def get_average_cell_state_value(CA1):
    cell4 = get_4_number_seq(CA1.cell_hist)
    idx_states = np.zeros((4, len(cell4[0, :])))

    for i in range(4):
        idx_states[i, :] = np.sum(cell4 == i + 1, axis=0) / len(cell4[:, 1])

    return idx_states


def get_stationary_dist(P, N):
    state = np.array([[1.0, 0.0, 0.0, 0.0]])
    for x in range(N):
        state = np.dot(state, P)

    return state


def get_unique_distances(CA1):
    D = np.round(CA1.relative_distance, 2)
    D_unique = np.unique(D)
    D_unique = np.zeros((2, len(D_unique)))
    D_unique[0, :] = np.unique(D)

    for c in range(len(D_unique[0, :])):
        idx = D[0, :] == D_unique[0, c]
        D_unique[1, c] = np.sum(idx)

    idx = np.zeros((len(D[:, 0]), len(D[0, :]), len(D_unique[0, :])), dtype=bool)

    for i in range(len(D_unique[0, :])):
        idx[:, :, i] = D == D_unique[0, i]

    return D_unique, idx


def get_AT_Hamiltonian(CA1, J):
    # Define Ising planes
    plane1 = copy.deepcopy(np.squeeze(CA1.cell_hist[:, 0, :]))
    plane2 = copy.deepcopy(np.squeeze(CA1.cell_hist[:, 1, :]))

    # Set change 0 spin state to -1
    plane1[plane1 == 0] = -1
    plane2[plane2 == 0] = -1

    # Calculate all dot products
    temp11 = np.swapaxes(np.repeat(plane1[:, :, np.newaxis], CA1.gridsize ** 2, axis=2), 1, 2)
    temp12 = np.swapaxes(temp11, 0, 1)

    temp21 = np.swapaxes(np.repeat(plane2[:, :, np.newaxis], CA1.gridsize ** 2, axis=2), 1, 2)
    temp22 = np.swapaxes(temp21, 0, 1)

    # Find neighbouring cells
    idx = np.round(CA1.relative_distance, 1) == 1
    neighbours = np.repeat(idx[:, :, np.newaxis], CA1.tmax, axis=2)

    # Calculate single elements of Hamiltonian
    H1 = np.sum(np.round(np.squeeze(np.sum(temp12 * temp11 * neighbours, axis=0))), axis=0)
    H2 = np.sum(np.round(np.squeeze(np.sum(temp22 * temp21 * neighbours, axis=0))), axis=0)
    H3 = np.sum(np.round(np.squeeze(np.sum(temp11 * temp12 * temp21 * temp22 * neighbours, axis=0))), axis=0)

    H4 = 0  # np.sum(plane1, axis=0)
    H5 = 0  # np.sum(plane2, axis=0)

    H = J[0] * H1 + J[1] * H2 + J[2] * H3 + J[3] * H4 + J[4] * H5

    H_all = np.zeros((5, len(H)))
    H_all[0, :] = H1
    H_all[1, :] = H2
    H_all[2, :] = H3
    H_all[3, :] = H4
    H_all[4, :] = H5

    return H, H_all


def get_AT_Hamiltonian_hist(hist_in, relative_distance, J, dist_val):
    N = len(hist_in[:, 0])
    tmax = len(hist_in[0, :])

    ''' 
    idx1 = hist_in == 1
    idx2 = hist_in == 2
    idx3 = hist_in == 3
    idx4 = hist_in == 4

    hist_in[idx1] = 4
    hist_in[idx2] = 1
    hist_in[idx3] = 2
    hist_in[idx4] = 3
    '''

    hist_temp = get_2_number_seq(hist_in)

    # Define Ising planes
    plane1 = copy.deepcopy((hist_temp[:, :, 0]))
    plane2 = copy.deepcopy((hist_temp[:, :, 1]))

    # Set change 0 spin state to -1
    plane1[plane1 == 0] = -1
    plane2[plane2 == 0] = -1

    # Calculate all dot products
    temp11 = np.swapaxes(np.repeat(plane1[:, :, np.newaxis], N, axis=2), 1, 2)
    temp12 = np.swapaxes(temp11, 0, 1)

    temp21 = np.swapaxes(np.repeat(plane2[:, :, np.newaxis], N, axis=2), 1, 2)
    temp22 = np.swapaxes(temp21, 0, 1)

    # Find neighbouring cells
    idx = np.round(relative_distance, 1) == dist_val
    neighbours = np.repeat(idx[:, :, np.newaxis], tmax, axis=2)

    # Calculate single elements of Hamiltonian
    H1 = J[0] * np.round(np.squeeze(np.sum(temp12 * temp11 * neighbours, axis=0)))
    H2 = J[1] * np.round(np.squeeze(np.sum(temp22 * temp21 * neighbours, axis=0)))
    H3 = J[2] * np.round(np.squeeze(np.sum(temp11 * temp12 * temp21 * temp22 * neighbours, axis=0)))
    H4 = J[3] * np.round(
        np.squeeze(np.sum((temp22 + temp21) * neighbours + (temp12 * temp11 + 1) * neighbours, axis=0)))

    H = np.sum(H1 + H2 + H3 + H4, axis=0)

    H_all = H1 + H2 + H3 + H4

    return H, H_all


def get_connected_areas(h, relative_distance):
    # Give size of the region to involve
    temp1 = np.round(relative_distance, 1) > 0
    temp2 = np.round(relative_distance, 1) <= 1
    temp = temp1 * temp2

    # Format True/False cells by repeating in time axis direction
    idx_neighbours = np.repeat(temp[:, :, np.newaxis], len(h[0, :]), axis=2)

    # Define matrix to store cluster labels and define background (h=-6 as 0)
    clusters = np.zeros_like(h)
    clusters[h != -6] = 1

    # Empty matrix to store cluster labels
    label_c = np.ones((len(h[0, :])))

    # Integer value of the number of neighbours
    nn = np.sum(temp[:, 0], axis=0)

    for j in range(3):
        for k in range(len(h[:, 0])):
            # get non zero pixel values
            idx1 = clusters[k, :] > 0

            # get values of the neighbouring cells
            idx_n = np.swapaxes(clusters[idx_neighbours[:, k, :].squeeze()].reshape(nn, len(h[0, :])), 0, 1)

            # find out if it belongs to new region or old
            idx_n[idx_n < 2] = 1
            idx_n_max = np.max(idx_n, axis=1)

            idx_old_region = idx_n_max > 1
            clusters[k, idx1 * idx_old_region] = idx_n_max[idx1 * idx_old_region]

            idx_new_region = idx_n_max == 1

            label_c[idx1 * idx_new_region] += 1
            clusters[k, idx1 * idx_new_region] = label_c[idx1 * idx_new_region]

    nclusters = np.zeros((len(h[0, :]), 1))
    for j in range(len(clusters[0, :])):
        nclusters[j] = len(np.unique(clusters[:, j], axis=0)) - 1

    return nclusters, clusters


def show_h_and_clusters(t0, nclusters, h, hist, positions, gz):
    fig, ax = plt.subplots()
    plt.subplot(3, 2, 1)
    plt.plot(nclusters)
    plt.title('Clusters')
    plt.xlabel('Time')
    plt.ylabel('Count')

    plt.subplot(3, 4, 3)
    plt.plot(np.sum(h, axis=0))
    plt.title('Total Hamiltonian')
    plt.ylabel('H')
    plt.xlabel('Time')

    plt.subplot(3, 4, 4)
    plt.plot(np.sum((hist[:, :] == 2) / gz ** 2, axis=0))
    plt.title('2-state cells')
    plt.xlabel('Time')
    plt.ylabel('Fraction')

    num_tsteps = 4

    cNorm = colors.Normalize(vmin=-6, vmax=6)  # normalise the colormap
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap='hot')  # map numbers to colors

    for t in range(num_tsteps):
        plt.subplot(3, num_tsteps, num_tsteps + 1 + t)
        plt.scatter(positions[:, 0], positions[:, 1], c=scalarMap.to_rgba(h[:, t0 + t]), edgecolors='black',
                    cmap=plt.get_cmap('hot'), s=40)
        plt.title('T=%i' % (t + t0))

    for t in range(num_tsteps):
        plt.subplot(3, num_tsteps, num_tsteps * 2 + 1 + t)
        plt.scatter(positions[:, 0], positions[:, 1], c=hist[:, t0 + t], cmap=plt.get_cmap('hot'),
                    edgecolors='black', s=40)
        plt.title('T=%i' % (t + t0))


def calculate_H_for_single(plane1, plane2, J, CA1):
    # Calculate all dot products
    temp11 = np.swapaxes(np.repeat(plane1[:, :, np.newaxis], CA1.gridsize ** 2, axis=2), 1, 2)
    temp12 = np.swapaxes(temp11, 0, 1)

    temp21 = np.swapaxes(np.repeat(plane2[:, :, np.newaxis], CA1.gridsize ** 2, axis=2), 1, 2)
    temp22 = np.swapaxes(temp21, 0, 1)

    # Find neighbouring cells
    idx = np.round(CA1.relative_distance, 1) == 1
    neighbours = np.repeat(idx[:, :, np.newaxis], 1, axis=2)

    # Calculate single elements of Hamiltonian
    H1 = J[0] * np.round(np.squeeze(np.sum(temp12 * temp11 * neighbours, axis=1)))
    H2 = J[1] * np.round(np.squeeze(np.sum(temp22 * temp21 * neighbours, axis=1)))
    H3 = J[2] * np.round(np.squeeze(np.sum(temp11 * temp12 * temp21 * temp22 * neighbours, axis=1)))

    H = (H1 + H2 + H3)

    return np.sum(H)


def compute_centroid(self):
    # number of cores to trace
    f_max = np.max(self.vortex_cores_labeled)

    x_com = np.ones((self.tmax, f_max))
    x_com[x_com == 1] = np.nan

    y_com = np.ones((self.tmax, f_max))
    y_com[y_com == 1] = np.nan

    # Set positions from CA object
    xx = self.positions[:, 0]
    yy = self.positions[:, 1]

    for j in range(f_max):
        # find indices of specific area
        c = self.vortex_cores_labeled == j + 1

        # compute center of mass
        thetaX = xx / np.max(xx) * np.pi * 2
        thetaY = yy / np.max(yy) * np.pi * 2

        alphaX = np.cos(thetaX)
        alphaY = np.cos(thetaY)

        betaX = np.sin(thetaX)
        betaY = np.sin(thetaY)

        count = np.sum(c, axis=0)
        count[count == 0] = 1

        thetaX_m = np.arctan2(-np.matmul(betaX, c) / count, -np.matmul(alphaX, c) / count) + np.pi
        thetaY_m = np.arctan2(-np.matmul(betaY, c) / count, -np.matmul(alphaY, c) / count) + np.pi

        x_com[:, j] = thetaX_m / (2 * np.pi) * np.max(xx)
        y_com[:, j] = thetaY_m / (2 * np.pi) * np.max(yy)

    return x_com, y_com


def initialize_lattice_from_excel(path, periodic_bc):  # Read input file
    position_matrix = pd.read_excel(path)

    # Format input data to remove nan padding
    position_matrix = position_matrix.to_numpy()
    position_matrix = position_matrix[~np.isnan(position_matrix).all(axis=1)]
    position_matrix = position_matrix[:, ~np.all(np.isnan(position_matrix), axis=0)]

    # Compute lattice coordinates
    xx, yy = np.meshgrid(np.arange(0, position_matrix.shape[1]), np.arange(0, position_matrix.shape[0]))

    # Define positions of each cell
    position_matrix[np.isnan(position_matrix)] = 0

    state = position_matrix[position_matrix > 0]

    x = (xx + 1) * (position_matrix > 0)
    y = (yy + 1) * (position_matrix > 0)

    positions = np.zeros((np.sum(x > 0), 2))
    positions[:, 0] = (x[x > 0] - 1)
    positions[:, 1] = (y[y > 0] - 1)

    # Flip y-axis to get correct configuration
    positions[:, 1] = np.abs(positions[:, 1] - np.max(positions[:, 1]))

    # Compute the relative distances and take periodic bc into account
    lx = np.max(positions[:, 0]) + 1
    ly = np.max(positions[:, 1]) + 1

    [x1, x2] = np.meshgrid(positions[:, 0], positions[:, 0])
    [y1, y2] = np.meshgrid(positions[:, 1], positions[:, 1])

    dx = np.mod(abs(x1 - x2), lx)
    dy = np.mod(abs(y1 - y2), ly)

    if periodic_bc[0] == 1:
        dx[dx > (lx - dx)] = lx - dx[dx > (lx - dx)]

    if periodic_bc[1] == 1:
        dy[dy > (ly - dy)] = ly - dy[dy > (ly - dy)]

    r = (dx ** 2 + dy ** 2) ** 0.5

    return positions, r, state


def get_contour_of_vortex_core(c, idx_label, relative_distance):
    hp_in = c == idx_label
    hp = c == idx_label
    idx = np.where(hp_in)

    # calculate neighbouring values and set tem all to true
    neighbours_idx = np.where(np.round(relative_distance[idx[0], :], 1) == 1)
    hp[neighbours_idx[1]] = True

    hp_in = hp_in * 1
    hp = hp * 1

    return hp - hp_in


def get_area_contour(c, idx_label, relative_distance):
    hp_in = c == idx_label
    hp = c == idx_label
    idx = np.where(hp_in)

    # calculate neighbouring values and set tem all to true
    neighbours_idx = np.where(np.round(relative_distance[idx[0], :], 1) == 1)
    hp[neighbours_idx[1]] = True

    hp_in = hp_in * 1
    hp = hp * 1

    return hp - hp_in


def get_contour_pos_relative_2_center(c_run, hist_run, idx_label, CA1):
    # compute values around vortex
    ring = get_area_contour(c_run, idx_label, CA1.relative_distance)

    idx_ring = np.where(ring == 1)
    x_r = CA1.positions[idx_ring, 0]
    y_r = CA1.positions[idx_ring, 1]
    c_r = hist_run[idx_ring]

    xx = CA1.positions[:, 0]
    yy = CA1.positions[:, 1]

    lx = np.max(CA1.positions[:, 0]) + CA1.positions[1, 0]
    ly = np.max(CA1.positions[:, 1]) + CA1.positions[1, 1]

    ring_bin = (ring == 1) * 1

    m_x_r, m_y_r = compute_center_of_mass(xx, yy, ring_bin)

    # Apply centroid algorithm
    y_hat = (y_r - m_y_r + ly / 2) % ly - ly / 2
    x_hat = (x_r - m_x_r + lx / 2) % lx - lx / 2

    return x_hat, y_hat, c_r, m_x_r, m_y_r


def get_vorticity(x_hat, y_hat, c_r):
    f_clock = np.zeros((4, 1))
    f_anti = np.zeros((4, 1))

    theta_r = np.arctan2(y_hat, x_hat) + np.pi
    idx_sort = np.argsort(theta_r)

    cwise = c_r[idx_sort]
    cwise_wrap = np.zeros((1, len(cwise[0, :]) + 1))
    cwise_wrap[0, :-1] = cwise
    cwise_wrap[0, len(cwise[0, :])] = cwise[0, 0]

    temp = cwise_wrap
    temp[temp == 4] = 5
    temp[temp == 3] = 4
    temp[temp == 5] = 3

    for j in range(4):
        temp = np.where(cwise == 1 + j)
        nn = cwise_wrap[0, temp[1] + 1]

        if j == 0 and len(nn) > 0:
            f_clock[j, 0] = (np.sum(nn == 1) + np.sum(nn == 2)) / len(nn)
            f_anti[j, 0] = (np.sum(nn == 1) + np.sum(nn == 3)) / len(nn)
        elif j == 1 and len(nn) > 0:
            f_clock[j, 0] = (np.sum(nn == 2) + np.sum(nn == 4)) / len(nn)
            f_anti[j, 0] = (np.sum(nn == 2) + np.sum(nn == 1)) / len(nn)
        elif j == 2 and len(nn) > 0:
            f_clock[j, 0] = (np.sum(nn == 3) + np.sum(nn == 1)) / len(nn)
            f_anti[j, 0] = (np.sum(nn == 3) + np.sum(nn == 4)) / len(nn)
        elif j == 3 and len(nn) > 0:
            f_clock[j, 0] = (np.sum(nn == 4) + np.sum(nn == 3)) / len(nn)
            f_anti[j, 0] = (np.sum(nn == 4) + np.sum(nn == 2)) / len(nn)

    return np.mean(f_clock, axis=0), np.mean(f_anti, axis=0)


def get_number_of_vortices(CA1, t_start, t_end, n_vortex, vortex_field, cell4):
    f_clock_tot = np.zeros((len(n_vortex), 1))
    f_anti_tot = np.zeros((len(n_vortex), 1))

    m_x_r_all = np.zeros((t_end - t_start, 2 + int(np.max(vortex_field)))) * np.nan
    m_y_r_all = np.zeros((t_end - t_start, 2 + int(np.max(vortex_field)))) * np.nan

    f_anti = np.zeros((t_end - t_start, 2 + int(np.max(vortex_field))))
    f_clock = np.zeros((t_end - t_start, 2 + int(np.max(vortex_field))))

    for p in range(t_end - t_start):
        # Set frame number
        fn = p + t_start

        # 4 states, h field and vortex field for frame
        hist_run = cell4[:, fn]
        c_run = vortex_field[:, fn]

        for idx_label in range(1, 1 + int(np.max(vortex_field[:, fn], axis=0))):
            x_hat, y_hat, c_r, m_x_r, m_y_r = get_contour_pos_relative_2_center(c_run, hist_run, idx_label, CA1)
            # print(np.sum(diff(c_r)))
            if np.any(x_hat > 0):
                m_x_r_all[p, idx_label] = m_x_r
                m_y_r_all[p, idx_label] = m_y_r
                f_clock[p, idx_label], f_anti[p, idx_label] = get_vorticity(x_hat, y_hat, c_r)

                if f_clock[p, idx_label] == 1 or f_anti[p, idx_label] == 1:
                    f_clock_tot[p, 0] += (f_clock[p, idx_label] >= 1) * 1
                    f_anti_tot[p, 0] += (f_anti[p, idx_label] >= 1) * 1
                else:  # post processing in case no integer vorticity is found in het contour
                    f_clock_tot[p, 0] += (f_clock[p, idx_label] >= 5 / 6) * 1
                    f_anti_tot[p, 0] += (f_anti[p, idx_label] >= 5 / 6) * 1

    return f_clock, f_anti, f_clock_tot, f_anti_tot, m_x_r_all, m_y_r_all


def compute_center_of_mass(xx, yy, idx_cells):
    # compute center of mass with periodic BC
    # See: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.151.8565&rep=rep1&type=pdf
    # 'Calculating Center of Mass in an Unbounded 2D Environment, L. Bai and D. Breen, journal of graphics tools'

    thetaX = xx / np.max(xx) * np.pi * 2
    thetaY = yy / np.max(yy) * np.pi * 2

    alphaX = np.cos(thetaX)
    alphaY = np.cos(thetaY)

    betaX = np.sin(thetaX)
    betaY = np.sin(thetaY)

    count = np.sum(idx_cells)

    if count > 0:
        thetaX_m = np.arctan2(-np.matmul(betaX, idx_cells) / count, -np.matmul(alphaX, idx_cells) / count) + np.pi
        thetaY_m = np.arctan2(-np.matmul(betaY, idx_cells) / count, -np.matmul(alphaY, idx_cells) / count) + np.pi
    else:
        thetaX_m = 0
        thetaY_m = 0

    x_com = thetaX_m / (2 * np.pi) * np.max(xx)
    y_com = thetaY_m / (2 * np.pi) * np.max(yy)

    return x_com, y_com


def apply_periodic_convolution(frame, kernel):
    r = sc.signal.convolve2d(frame, kernel, boundary='wrap', mode='same')
    r[r <= -np.pi] = r[r <= -np.pi] + 2 * np.pi
    r[np.pi <= r] = r[np.pi <= r] - 2 * np.pi

    return r


def get_contour_kernel(Nkernel):
    tot_num_kernels = 4*(Nkernel-1)

    K = np.zeros((Nkernel, Nkernel, tot_num_kernels))
    top_vect = np.zeros((1, Nkernel))
    top_vect[0, 0] = 1
    top_vect[0, 1] = -1

    # fill top row
    for i in range(Nkernel-1):
        K[0, i:(i+2), i] = np.array([-1, 1])

    # fill right column
    for i in range(Nkernel-1):
        K[i:(i+2), -1, i+Nkernel-1] = np.array([-1, 1])

    # fill bottom row
    for i in range(Nkernel-1):
        K[-1, i:(i+2), i+2*(Nkernel-1)] = np.array([1, -1])

    # fill left column
    for i in range(Nkernel-1):
        K[i:(i+2), 0, i+3*(Nkernel-1)] = np.array([1, -1])

        return K


#### Hamiltonian functions
def show_spin_states_of_configuration(hist_in, h_in, pos, t):
    U = np.zeros_like(hist_in, dtype=np.double)
    V = np.zeros_like(hist_in, dtype=np.double)

    U[hist_in == 1] = np.cos(0)
    V[hist_in == 1] = np.sin(0)
    U[hist_in == 2] = np.cos(np.pi / 2)
    V[hist_in == 2] = np.sin(np.pi / 2)
    U[hist_in == 3] = np.cos(np.pi * 3 / 2)
    V[hist_in == 3] = np.sin(np.pi * 3 / 2)
    U[hist_in == 4] = np.cos(np.pi)
    V[hist_in == 4] = np.sin(np.pi)

    g = 2
    fig, ax = plt.subplots()
    ax.quiver(pos[:, 0], pos[:, 1], U * g, V * g)

    ax.scatter(pos[:, 0], pos[:, 1], c=h_in, edgecolors='black', cmap=plt.get_cmap('hot'), s=80)
    plt.title('Time step: ' + str(t))
    plt.show()

    return ax


def show_hamiltonians(input):
    plt.figure()
    plt.plot(input.Hxy_adjusted, label='Adjust Hamiltonian')
    plt.plot(input.Hxy_classic, label='Classic XY Hamiltonian')
    plt.xlabel('Time')
    plt.ylabel('H')
    plt.title('Normalized Hamiltonian for trajectory')
    plt.legend()
