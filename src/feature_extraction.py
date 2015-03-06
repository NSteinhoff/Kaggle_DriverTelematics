__author__ = 'nikosteinhoff'

import numpy as np
import src.data_reading as data_reading

drivers = data_reading.get_drivers()
trips_per_driver = data_reading.get_all_filenames(['1', '2'])


def build_driver_data(driver):
    trips = trips_per_driver[driver]

    driver_data = np.zeros((1, 1))
    for trip in trips:
        # extract features
        trip_features = extract_trip_features(driver, trip)

        # append row to array
        if driver_data.size == 1:
            driver_data = np.copy(trip_features)
        else:
            driver_data = np.vstack((driver_data, trip_features))

    return driver_data


def extract_trip_features(driver, trip):
    trip_data = data_reading.load_trip_data(driver, trip)

    # TODO All the data transformation stuff here

    # A single row of features per trip
    features = np.arange(10)

    return features






if __name__ == '__main__':
    print("Running as main")

    data = build_driver_data('1')
    print(data.shape)
    print(data)
