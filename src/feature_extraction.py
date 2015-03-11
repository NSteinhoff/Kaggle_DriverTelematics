__author__ = 'nikosteinhoff'

import numpy as np
from scipy import signal
from src import file_handling
from src import classification_model
import random
import matplotlib.pyplot as plt
from multiprocessing import Process, Pipe
import time


def calculate_driver(driver, mp=False):
    print("Calculating driver {0}".format(driver))

    data = build_data_set(driver, mp=mp)

    probabilities = classification_model.classify_data(data)

    sorted_probabilities = probabilities[probabilities[:, 1].argsort()]

    calibration = np.linspace(0, 100, 200)
    calibrated_probabilities = np.column_stack((sorted_probabilities, calibration))

    sorted_calibrated_probabilities = calibrated_probabilities[calibrated_probabilities[:, 0].argsort()]

    driver_results = np.column_stack((np.ones((sorted_calibrated_probabilities.shape[0], 1))*driver, sorted_calibrated_probabilities))
    return driver_results


def build_data_set(driver, mp=False):
    if mp:
        # Open pipes
        receiver_driver_data, sender_driver_data = Pipe(duplex=False)
        receiver_ref_data, sender_ref_data = Pipe(duplex=False)

        # Start processes
        p_driver_data = Process(target=piped_process, args=(sender_driver_data, build_driver_data, driver))
        p_ref_data = Process(target=piped_process, args=(sender_ref_data, build_reference_data, 200, 1, driver))
        p_driver_data.start()
        p_ref_data.start()

        # Retrieve data from pipes
        driver_data = receiver_driver_data.recv()
        ref_data = receiver_ref_data.recv()

        # Exit processes
        p_driver_data.join()
        p_ref_data.join()
    else:
        driver_data = build_driver_data(driver)
        ref_data = build_reference_data(200, 1, exclude=driver)

    driver_data = np.column_stack((np.ones((driver_data.shape[0], 1), dtype=float), driver_data))  # Add label
    ref_data = np.column_stack((np.zeros((ref_data.shape[0], 1), dtype=float), ref_data))  # Add label


    complete_data = np.vstack((driver_data, ref_data))
    print("Complete data set for driver {0} --->> {1}".format(driver, complete_data.shape))

    return complete_data


def piped_process(pipe, function, *args):
    return_value = function(*args)

    pipe.send(return_value)
    pipe.close()


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
    col_quartiles = np.percentile(transformed_data[:, 4:], [25, 75], axis=0)
    col_IQR = col_quartiles[1] - col_quartiles[0]

    # A single row of features per trip
    features = np.hstack((trip, duration, length, col_means, col_std, col_IQR))

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

    convolve_N = 3
    convolve_array = np.ones((convolve_N, ))/convolve_N
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

        radiants = radiants_between(current_vector, previous_vector)

        directional_changes.append(radiants)

    return directional_changes


def radiants_between(v1, v2):
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
    test_data = build_data_set(1, mp=True)
    print(test_data.shape)
    print("Elapsed = {0:.2f}".format(time.time() - start_time))
    print("done!")