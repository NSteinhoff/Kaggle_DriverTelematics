__author__ = 'nikosteinhoff'

import numpy as np
from scipy import signal
from src import file_handling
import random
import matplotlib.pyplot as plt
from multiprocessing import Process, Pipe
import time


def build_data_set(driver):
    path = os.path.join(file_handling._data_directory,'drivers', str(driver), 'data_set.csv')

    driver_data = build_driver_data(driver)
    ref_data = build_reference_data(200, 1, exclude=driver)

    driver_data = np.column_stack((np.ones((driver_data.shape[0], 1), dtype=float), driver_data))  # Add label
    ref_data = np.column_stack((np.zeros((ref_data.shape[0], 1), dtype=float), ref_data))  # Add label

    complete_data = np.vstack((driver_data, ref_data))

    np.savetxt(path, complete_data, delimiter=',')

    return complete_data


def build_driver_data(driver, sample_size=None):
    trips = file_handling.get_trips(driver)

    if sample_size is not None and type(sample_size) == int:
        trips = random.sample(trips, sample_size)

    driver_data = np.zeros((1, 1), dtype=float)
    for trip in trips:
        # extract features
        trip_features = extract_trip_features(driver, trip)

        # append row to array
        if driver_data.size == 1:
            driver_data = np.copy(trip_features)
        else:
            driver_data = np.vstack((driver_data, trip_features))

    return driver_data


def build_reference_data(n_drivers=200, n_trips=1, exclude=None):
    drivers = [f for f in file_handling.get_drivers() if f is not exclude]
    random.seed(123456)
    drivers_sample = random.sample(drivers, n_drivers)

    reference_data = np.zeros((1, 1), dtype=float)
    for driver in drivers_sample:
        driver_data = build_driver_data(driver, n_trips)

        if reference_data.size == 1:
            reference_data = np.copy(driver_data)
        else:
            reference_data = np.vstack((reference_data, driver_data))

    return reference_data


def extract_trip_features(driver, trip):
    trip_data = file_handling.load_trip_data(driver, trip)
    transformed_data = transform_data(trip_data)

    # Duration
    duration = transformed_data.shape[0]

    # Length
    length = transformed_data[:, 4].sum()

    # Means
    col_means = transformed_data[:, 4:].mean(axis=0)

    # Standard deviations
    col_std = transformed_data[:, 4:].std(axis=0)

    # Interquartile ranges
    col_percentiles = np.percentile(transformed_data[:, 4:], [10, 25, 50, 75, 90], axis=0)
    col_iqr = col_percentiles[3] - col_percentiles[1]
    col_10_90_range = col_percentiles[0] - col_percentiles[4]
    col_med = col_percentiles[2]

    # A single row of features per trip
    features = np.hstack((trip, duration, length, col_means, col_std, col_iqr, col_10_90_range, col_med))
    descriptions = [
                    'duration', 'length',
                    'mean_velocity', 'mean_acceleration', 'mean_directional_change',
                    'mean_velocity*dir_change', 'mean_velocity*acceleration', 'mean_acceleration*dir_change',
                    'std_velocity', 'std_acceleration', 'std_directional_change',
                    'std_velocity*dir_change', 'std_velocity*acceleration', 'std_acceleration*dir_change', 
                    'iqr_velocity', 'iqr_acceleration', 'iqr_directional_change',
                    'iqr_velocity*dir_change',  'iqr_velocity*acceleration', 'iqr_acceleration*dir_change',
                    '90pcr_velocity', '90pcr_acceleration', '90pcr_directional_change',
                    '90pcr_velocity*dir_change',  '90pcr_velocity*acceleration', '90pcr_acceleration*dir_change',
                    'median_velocity', 'median_acceleration', 'median_directional_change',
                    'median_velocity*dir_change', 'median_velocity*acceleration', 'median_acceleration*dir_change'
                    ]

    return features


def transform_data(raw_data, plot=False):
    temp_data = np.copy(raw_data)
    ix_x = 0
    ix_y = 1

    # X and Y changes from measuring period t to t+1
    x_change = [0]
    y_change = [0]
    for i in range(1, temp_data.shape[0]):
        x_change.append(temp_data[i, ix_x] - temp_data[i-1, ix_x])
        y_change.append(temp_data[i, ix_y] - temp_data[i-1, ix_y])

    # Removing spikes and smoothing
    x_change_no_spikes = signal.medfilt(x_change)
    y_change_no_spikes = signal.medfilt(y_change)

    convolve_n = 3
    convolve_array = np.ones((convolve_n, ))/convolve_n
    x_change_smooth = signal.convolve(x_change_no_spikes, convolve_array, mode='same')
    y_change_smooth = signal.convolve(y_change_no_spikes, convolve_array, mode='same')

    if plot:
        period = range(len(x_change))
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(period, x_change, 'r-', period, x_change_smooth, 'g-')
        plt.ylabel("x_change")

        plt.subplot(2, 1, 2)
        plt.plot(period, y_change, 'r-', period, y_change_smooth, 'g-')
        plt.ylabel("y_change")

        plt.show()

    temp_data = np.column_stack((temp_data, x_change_smooth, y_change_smooth))
    ix_x_change = 2
    ix_y_change = 3

    # Velocity
    velocity = calculate_speed(temp_data, ix_x_change, ix_y_change)
    temp_data = np.column_stack((temp_data, velocity))
    ix_velocity = 4

    # Acceleration
    acceleration = [0]
    for i in range(1, temp_data.shape[0]):
        acceleration.append(temp_data[i, ix_velocity] - temp_data[i-1, ix_velocity])
    temp_data = np.column_stack((temp_data, acceleration))
    ix_acceleration = 5

    # Directional changes
    directional_changes = calculation_direction_change(temp_data, ix_x_change, ix_y_change)
    temp_data = np.column_stack((temp_data, directional_changes))
    ix_direction_changes = 6

    # Interactions between speed, directional changes and acceleration
    speed_times_turn = []
    speed_times_acceleration = []
    acceleration_times_turn = []
    for row in temp_data:
        speed_times_turn.append(row[ix_velocity] * row[ix_direction_changes])
        speed_times_acceleration.append(row[ix_velocity] * row[ix_acceleration])
        acceleration_times_turn.append(row[ix_acceleration] * row[ix_direction_changes])
    temp_data = np.column_stack((temp_data, speed_times_turn, speed_times_acceleration, acceleration_times_turn))

    transformed_data = np.copy(temp_data)

    return transformed_data


def calculate_speed(data, index_1, index_2):
    speeds = []
    for i in range(data.shape[0]):
        x = data[i, index_1]
        y = data[i, index_2]

        norm = pow(x, 2) + pow(y, 2)

        speed = np.sqrt(norm)

        speeds.append(speed)

    return speeds


def calculation_direction_change(data, index_1, index_2):
    directional_changes = [0]

    for i in range(1, data.shape[0]):
        current_vector = [data[i, index_1], data[i, index_2]]
        previous_vector = [data[i-1, index_1], data[i-1, index_2]]

        radians = radians_between(current_vector, previous_vector)

        directional_changes.append(radians)

    return directional_changes


def radians_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            angle_between((1, 0, 0), (1, 0, 0))
            0.0
            angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if all([element == 0 for element in v1]) or all([element == 0 for element in v2]):
        return 0

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1, 1))

    return angle


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


if __name__ == '__main__':
    start_time = time.time()
    test_data, feature_descriptions = build_data_set(1)
    print(test_data.shape)
    print("Elapsed = {0:.2f}".format(time.time() - start_time))
    print("done!")