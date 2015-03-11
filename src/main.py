__author__ = "nikosteinhoff"
__email__ = "niko.steinhoff@gmail.com"

import numpy as np
from src import file_handling
from src import feature_extraction
from src import classification_model
from multiprocessing import Pool

import time


def main(test=False):
    print("This is main()")
    drivers = file_handling.get_drivers()

    file_handling.write_to_submission_file("driver_trip,prob", overwrite=True, test=test)
    start_time = time.time()
    if test:
        drivers = drivers[:24]

    with Pool() as p:
        res = p.map(classification_model.calculate_driver, drivers)
    results = []
    for item in res:
        for row in item:
            results.append("{0}_{1},{2:.6f}".format(int(row[0]), int(row[1]), row[3]))

    print("\n+-----------------+")
    print("Total time for {0} drivers   -----> {1:.1f}  sec".format(len(drivers), time.time()-start_time))
    print("Per driver                 -----> {0:.1f} sec".format((time.time()-start_time)/len(drivers)))
    print("+-----------------+")

    for item in results:
        file_handling.write_to_submission_file(item, test=test)


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=2)
    main(test=False)