__author__ = "nikosteinhoff"
__email__ = "niko.steinhoff@gmail.com"

import numpy as np
import src.file_handling as file_handling
import src.feature_extraction as feature_extraction
import src.classification_model as classification_model


def main(test=False):
    print("This is main()")
    drivers = file_handling.get_drivers()

    file_handling.write_to_submission_file("driver_trip,prob\n", overwrite=True)

    if test:
        drivers = drivers[:3]

    calibration = np.linspace(0, 100, 200)

    count = 0
    for driver in drivers:
        count += 1
        print("Calculating {0}/{1}".format(count, len(drivers)))
        data = feature_extraction.build_data_set(driver)
        print(data)

        probabilities = classification_model.classify_data(data)

        sorted_probablities = probabilities[probabilities[:, 1].argsort()]

        calibrated_probabilities = np.column_stack((sorted_probablities, calibration))

        print(calibrated_probabilities)

        for element in calibrated_probabilities:
            file_handling.write_to_submission_file("{0}_{1},{2}\n".format(driver, element[0], element[2]))




if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=2)
    main(test=False)