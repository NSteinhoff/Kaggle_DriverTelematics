import numpy as np
import pandas as pd
import os


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
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1, 1))

    return angle


def write_to_file(driver, line):
    path = "/Volumes/DATA/Data/Kaggle/AxaDriverTelematics/summary_driver_{0}.csv".format(str(driver))
    with open(path, 'a') as file:
        file.write(line + "\n")
    print("Finished writing to file {0}".format(path))


def create_file_with_header(driver, header):
    path = "/Volumes/DATA/Data/Kaggle/AxaDriverTelematics/summary_driver_{0}.csv".format(str(driver))

    if not os.path.isfile(path):
        with open(path, 'w') as file:
            file.write(header + "\n")
        print("Finished creating file {0}".format(path))


def get_trip_summaries(driver=1, trip=1, all = False):
    # Trip data import

    folder = '/Volumes/DATA/Data/Kaggle/AxaDriverTelematics/drivers/{0}/'.format(driver)

    files = []
    if all:
        files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    else:
        files.append('{0}.csv'.format(trip))

    count = 0
    for file in files:
        count += 1
        print("Current file: {0}".format(file))
        print("{0} / {1}".format(count, len(files)))

        trip = int(file[:-4])
        print("Trip {0}".format(trip))

        path = '{0}/{1}'.format(folder, file)

        data_raw = pd.read_csv(path)

        data = data_raw.copy()
        data['change_x'] = data_raw.x - data_raw.x.shift(1)
        data['change_y'] = data_raw.y - data_raw.y.shift(1)

        # Speed and acceleration
        data['speed'] = np.sqrt(pow(data.change_x, 2) + pow(data.change_y, 2))
        data['acceleration'] = data.speed - data.speed.shift(1)

        # Directional change
        data['direction_change'] = np.nan

        index_change_x = data.columns.get_loc('change_x')
        index_change_y = data.columns.get_loc('change_y')
        index_direction_change = data.columns.get_loc('direction_change')

        for i in range(1, data.shape[0]):
            current_vector = [data.iloc[i, index_change_x], data.iloc[i, index_change_y]]
            previous_vector = [data.iloc[i-1, index_change_x], data.iloc[i-1, index_change_y]]
            data.iloc[i, index_direction_change] = radiants_between(current_vector, previous_vector)

        data['speed_X_acceleration'] = data.speed * data.acceleration
        data['speed_X_direction_change'] = data.speed * data.direction_change
        data['acceleration_X_direction_change'] = data.acceleration * data.direction_change
        data['speed_X_acceleration_X_direction_change'] = data.speed * data.acceleration * data.direction_change

        # Removing data points with very low speed (standing still).
        # Strong directional changes with very little actual movements
        # could be due to GPS readings varying slightly without the car moving at all.
        #
        # Cutoff value???

        data = data[data.speed > .25]

        # Summarizing trip
        means = data.loc[:, 'speed':].mean()
        std  = data.loc[:, 'speed':].std()

        # Extend column names
        new_index_mean = []
        for name in means.index:
            new_index_mean.append('mean_{0}'.format(name))

        means.index = new_index_mean

        new_index_std = []
        for name in std.index:
            new_index_std.append('std_{0}'.format(name))

        std.index = new_index_std

        # Create complete summary
        summary_row = [driver, trip]
        summary_row.extend(means.values)
        summary_row.extend(std.values)

        # Create blank dataframe
        summary_columns = ['driver', 'trip']
        summary_columns.extend(means.index.tolist())
        summary_columns.extend(std.index.tolist())
        summary = pd.DataFrame(columns = summary_columns)

        # Add trip stats
        summary.loc[trip] = summary_row

        summary_line = ",".join(map(str, summary_row))

        create_file_with_header(driver, ",".join(summary_columns))

        write_to_file(driver, summary_line)


if __name__ == "__main__":
    get_trip_summaries(driver=1, all=True)