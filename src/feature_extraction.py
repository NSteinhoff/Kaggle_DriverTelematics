__author__ = 'nikosteinhoff'

import numpy as np
import src.file_handling as file_handling
import random

random.seed(123456)


def build_data_set(driver):
    driver_data = build_driver_data(driver)
    driver_data = np.column_stack((np.ones((driver_data.shape[0], 1), dtype=float), driver_data))  # Add label
    print("driver_data shape = {0}".format(driver_data.shape))

    ref_data = build_reference_data(exclude=driver)
    ref_data = np.column_stack((np.zeros((ref_data.shape[0], 1), dtype=float), ref_data))  # Add label
    print("ref_data shape = {0}".format(ref_data.shape))

    print("Complete data set for driver {0}".format(driver))
    complete_data = np.vstack((driver_data, ref_data))
    print(complete_data.shape)

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

    col_percentiles = np.percentile(transformed_data[:, 4:], range(25, 100, 25), axis=0)

    # TODO All the data transformation stuff here

    # A single row of features per trip
    trip_number = int(trip[:-4])
    features = np.hstack((np.array(trip_number), col_percentiles.ravel('F')))

    return features


def transform_data(raw_data):
    temp_data = np.copy(raw_data)
    ix_x = 0
    ix_y = 1

    # X and Y changes from measuring period t to t+1
    x_change = [0]
    y_change = [0]
    for i in range(1, temp_data.shape[0]):
        x_change.append(temp_data[i, ix_x] - temp_data[i-1, ix_x])
        y_change.append(temp_data[i, ix_y] - temp_data[i-1, ix_y])

    temp_data = np.column_stack((temp_data, x_change, y_change))
    ix_x_change = 2
    ix_y_change = 3

    # Speed
    speed = calculate_speed(temp_data, ix_x_change, ix_y_change)
    temp_data = np.column_stack((temp_data, speed))
    ix_speed = 4

    # Acceleration
    acceleration = [0]
    for i in range(1, temp_data.shape[0]):
        acceleration.append(temp_data[i, ix_speed] - temp_data[i-1, ix_speed])
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


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


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


if __name__ == '__main__':
    print("Running as main")
    data = build_data_set('1')

    np.set_printoptions(suppress=True, precision=2)
    print(data)
