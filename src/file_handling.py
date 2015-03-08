__author__ = 'nikosteinhoff'

import numpy as np
import os


# Global vars:
_data_directory = '/Users/nikosteinhoff/Data/Kaggle/AxaDriverTelematics'


def get_drivers(data_directory=None):
    if not data_directory:
        data_directory = os.path.join(_data_directory, 'drivers')

    drivers = [f for f in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, f))]

    return drivers


def get_trips(driver, data_directory=None):
    if not data_directory:
        data_directory = os.path.join(_data_directory, 'drivers')

    trip_directory = os.path.join(data_directory, str(driver))

    trips = [f for f in os.listdir(trip_directory)
             if os.path.isfile(os.path.join(trip_directory, f))
             and f.endswith('.csv')]

    return trips


def load_trip_data(driver, trip, data_directory=None):
    if not data_directory:
        data_directory = os.path.join(_data_directory, 'drivers')

    path = os.path.join(data_directory, driver, trip)

    data = np.genfromtxt(path, skip_header=1, delimiter=',')

    return data


def write_submission_file(data_directory=None):
    if not data_directory:
        data_directory = _data_directory

    file_path = os.path.join(data_directory, 'submission_file.csv')

    drivers = get_drivers()
    trips = range(1, 201)
    prob = 1

    with open(file_path, 'w') as file:
        file.write('driver_trip,prob\n')

        for driver in drivers:
            for trip in trips:
                file.write('{0}_{1},{2}\n'.format(driver, trip, prob))