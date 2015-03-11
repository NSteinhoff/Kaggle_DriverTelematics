__author__ = "nikosteinhoff"
__email__ = "niko.steinhoff@gmail.com"

import numpy as np
import src.file_handling as file_handling
import src.feature_extraction as feature_extraction
import src.classification_model as classification_model
import time


def main(test=False):
    print("This is main()")
    drivers = file_handling.get_drivers()

    file_handling.write_to_submission_file("driver_trip,prob\n", overwrite=True)

    if test:
        drivers = drivers[:3]

    calibration = np.linspace(0, 100, 200)

    count = 0
    start_time = time.time()
    for driver in drivers:
        count += 1
        print("Calculating {0}/{1}".format(count, len(drivers)))

        data = feature_extraction.build_data_set(driver, mp=True)
        print(data)

        probabilities = classification_model.classify_data(data)

        sorted_probabilities = probabilities[probabilities[:, 1].argsort()]

        calibrated_probabilities = np.column_stack((sorted_probabilities, calibration))

        sorted_calibrated_probabilities = calibrated_probabilities[calibrated_probabilities[:, 0].argsort()]

        print(sorted_calibrated_probabilities)

        for element in sorted_calibrated_probabilities:
            file_handling.write_to_submission_file("{0}_{1},{2:.6f}\n".format(driver, int(element[0]), element[2]))

        elapsed_time = time.time() - start_time
        time_per_driver = elapsed_time / count
        total_time_estimate = time_per_driver * len(drivers)
        finish_time = start_time + total_time_estimate

        print("+-----------------+")
        print("Finished driver {0}/{1}".format(count, len(drivers)))
        print("Started at: {0}".format(time.ctime(start_time)))
        print("Seconds per driver: {0:.2f}".format(time_per_driver))
        print("Done at: {0}".format(time.ctime(finish_time)))
        print("+-----------------+")



if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=2)
    main(test=False)