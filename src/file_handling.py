__author__ = 'nikosteinhoff'

import numpy as np
import os


# Global vars:
_data_directories = ['/Users/nikosteinhoff/Data/Kaggle/AxaDriverTelematics', '/home/nikosteinhoff/Data/Kaggle/AxaDriverTelematics']
for dir in _data_directories:
    if os.path.isdir(dir):
        _data_directory = dir
print(_data_directory)


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


def write_to_submission_file(line, overwrite=False, data_directory=None):
    if not data_directory:
        data_directory = _data_directory

    file_path = os.path.join(data_directory, 'submission_file.csv')

    if overwrite:
        mode = 'w'
    else:
        mode = 'a'

    with open(file_path, mode) as file:
        file.write(line)