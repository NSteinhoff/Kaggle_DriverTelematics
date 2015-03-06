__author__ = 'nikosteinhoff'

import numpy as np
import os


# Global vars:
_data_directory = '/Users/nikosteinhoff/Data/Kaggle/AxaDriverTelematics/drivers'


def get_drivers(data_directory=None):
    if not data_directory:
        data_directory = _data_directory

    drivers = [f for f in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, f))]

    return drivers


def get_trips(driver, data_directory=None):
    if not data_directory:
        data_directory = _data_directory

    trip_directory = os.path.join(data_directory, str(driver))

    trips = [f for f in os.listdir(trip_directory)
             if os.path.isfile(os.path.join(trip_directory, f))
             and f.endswith('.csv')]

    return trips


def get_all_filenames(drivers, data_directory=None):
    if not data_directory:
        data_directory = _data_directory

    filenames_by_drivers = {}
    for driver in drivers:
        trips = get_trips(driver, data_directory)
        filenames_by_drivers[driver] = trips

    return filenames_by_drivers


def load_trip_data(driver, trip, data_directory=None):
    if not data_directory:
        data_directory = _data_directory

    path = os.path.join(data_directory, driver, trip)

    data = np.genfromtxt(path, skip_header=1, delimiter=',')

    return data