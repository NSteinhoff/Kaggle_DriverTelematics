__author__ = "nikosteinhoff"
__email__ = "niko.steinhoff@gmail.com"

import numpy as np
from src import file_handling
from src import feature_extraction
from src import multiprocessing_helpers

import time


def main(test=False, mp=False):
    print("This is main()")
    drivers = file_handling.get_drivers()

    file_handling.write_to_submission_file("driver_trip,prob", overwrite=True)

    count = 0
    start_time = time.time()

    if test:
        drivers = drivers[:6]

    if mp:
        process_manager = multiprocessing_helpers.ProcessManager(feature_extraction.calculate_driver)
        for driver in drivers:
            process_manager.in_queue.put(driver)

        process_manager.start_processing()

        results = []
        while not process_manager.out_queue.empty():
            next_results = process_manager.out_queue.get()

            for item in next_results:
                results.append("{0}_{1},{2:.6f}".format(int(item[0]), int(item[1]), item[3]))

    else:
        results = []
        for driver in drivers:
            count += 1
            print("Calculating {0}/{1}".format(count, len(drivers)))

            driver_results = feature_extraction.calculate_driver(driver, True)

            for element in driver_results:
                results.append("{0}_{1},{2:.6f}".format(int(element[0]), int(element[1]), element[3]))

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

    print("\n+-----------------+")
    print("Total time for {0} drivers   -----> {1:.1f}  sec".format(len(drivers), time.time()-start_time))
    print("Per driver                 -----> {0:.1f} sec".format((time.time()-start_time)/len(drivers)))
    print("+-----------------+")

    for item in results:
        file_handling.write_to_submission_file(item)


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=2)
    main(test=True, mp=True)