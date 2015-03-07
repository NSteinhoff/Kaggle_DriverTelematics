__author__ = 'nikosteinhoff'

import numpy as np
import src.data_reading as data_reading
import random

random.seed(123456)


def build_data_set(driver):

    data = build_driver_data('1', 5)
    data = np.column_stack((np.ones((data.shape[0], 1), dtype=float), data))  # Add label
    print("driver_data shape = {0}".format(data.shape))

    ref_data = build_reference_data(5, 1)
    ref_data = np.column_stack((np.zeros((ref_data.shape[0], 1), dtype=float), ref_data))  # Add label
    print("ref_data shape = {0}".format(ref_data.shape))

    print("Complete data set for driver {0}".format(driver))
    complete_data = np.vstack((data, ref_data))
    print(complete_data.shape)
    print(complete_data)


def build_driver_data(driver, sample_size=None):
    trips = data_reading.get_trips(driver)

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


def build_reference_data(n_drivers=200, n_trips=1):
    drivers = data_reading.get_drivers()
    drivers_sample = random.sample(drivers, n_drivers)

    reference_data = np.zeros((1, 1), dtype=float)
    for driver in drivers_sample:
        driver_data = build_driver_data(driver, n_trips)

        if reference_data.size ==1:
            reference_data = np.copy(driver_data)
        else:
            reference_data = np.vstack((reference_data, driver_data))

    return reference_data


def extract_trip_features(driver, trip):
    trip_data = data_reading.load_trip_data(driver, trip)

    # TODO All the data transformation stuff here

    # A single row of features per trip
    trip_number = int(trip[:-4])
    features = np.hstack((np.array(trip_number), np.arange(10)))  # TODO These are dummy values

    return features


if __name__ == '__main__':
    print("Running as main")
    build_data_set('1')

