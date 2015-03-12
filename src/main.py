__author__ = "nikosteinhoff"
__email__ = "niko.steinhoff@gmail.com"

import numpy as np
from src import file_handling
from src import classification_model
import multiprocessing
from multiprocessing import Pool

import time


def estimate_duration(drivers):
    start_time = time.time()
    timing_sample = drivers[:multiprocessing.cpu_count() * 3]
    with Pool() as test_p:
        test_p.map(classification_model.calculate_driver, timing_sample)
    avg_time = (time.time() - start_time) / len(timing_sample)
    estimated_total_time = avg_time * len(drivers)
    estimated_completion_time = start_time + estimated_total_time
    print("\n+-----------------+")
    print("Average time per driver {0:.1f}".format(avg_time))
    print("Estimated completion time: {0}".format(time.ctime(estimated_completion_time)))
    print("+-----------------+\n")


def main(test=False):
    print("This is main()")
    start_time = time.time()
    drivers = file_handling.get_drivers()
    file_handling.write_to_submission_file("driver_trip,prob", overwrite=True, test=test)
    if test:
        drivers = drivers[:8]
    else:
        estimate_duration(drivers)

    with Pool() as p:
        results = p.map(classification_model.calculate_driver, drivers)

    for item in results:
        for row in item:
            file_handling.write_to_submission_file("{0}_{1},{2:.6f}".format(
                int(row[0]), int(row[1]), row[3]), test=test)

    total_time = time.time()-start_time
    print("\n+-----------------+")
    print("Total time for {0} drivers   -----> {1:.1f}  sec".format(len(drivers), total_time))
    print("Per driver                 -----> {0:.1f} sec".format(total_time/len(drivers)))
    print("+-----------------+")


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=2)
    main(test=False)