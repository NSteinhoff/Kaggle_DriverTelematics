__author__ = 'nikosteinhoff'

import numpy as np
import os
import time

# Global vars:
_data_directories = ['/Users/nikosteinhoff/Data/Kaggle/AxaDriverTelematics',
                     '/home/nikosteinhoff/Data/Kaggle/AxaDriverTelematics']
_data_directory = ""
for directory in _data_directories:
    if os.path.isdir(directory):
        _data_directory = directory
print(_data_directory)

file_creation_time = time.time()


def get_drivers(data_directory=None):
    if not data_directory:
        data_directory = os.path.join(_data_directory, 'drivers')

    drivers = [int(f) for f in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, f))]

    return sorted(drivers)


def get_trips(driver, data_directory=None):
    if not data_directory:
        data_directory = os.path.join(_data_directory, 'drivers')

    trip_directory = os.path.join(data_directory, str(driver))

    trips = [int(f[:-4]) for f in os.listdir(trip_directory)
             if os.path.isfile(os.path.join(trip_directory, f))
             and f.endswith('.csv')]

    return sorted(trips)


def load_trip_data(driver, trip, data_directory=None):
    if not data_directory:
        data_directory = os.path.join(_data_directory, 'drivers')

    path = os.path.join(data_directory, str(driver), '{0}.csv'.format(trip))

    data = np.genfromtxt(path, skip_header=1, delimiter=',')

    return data


def write_to_submission_file(line, overwrite=False, test=False, data_directory=None):
    if not data_directory:
        data_directory = _data_directory

    if test:
        file_path = os.path.join(data_directory, 'submission_file_test.csv')
    else:
        file_path = os.path.join(data_directory, 'submission_file_{0}.csv'.format(int(file_creation_time)))

    if overwrite:
        mode = 'w'
    else:
        mode = 'a'

    with open(file_path, mode) as file:
        file.write(line+'\n')


def write_to_model_frequencies_file(line, overwrite=False, test=False, data_directory=None):
    if not data_directory:
        data_directory = _data_directory

    if test:
        file_path = os.path.join(data_directory, 'model_frequencies_test.csv')
    else:
        file_path = os.path.join(data_directory, 'model_frequencies_{0}.csv'.format(int(file_creation_time)))

    if overwrite:
        mode = 'w'
    else:
        mode = 'a'

    with open(file_path, mode) as file:
        file.write(line+'\n')

if __name__ == '__main__':
    drivers = get_drivers()
    trips = get_trips(1)
    trip_data = load_trip_data(1, 1)