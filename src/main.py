__author__ = "nikosteinhoff"
__email__ = "niko.steinhoff@gmail.com"

import numpy as np
from src import file_handling
from src import classification_model
import multiprocessing
from multiprocessing import Pool
from functools import partial

import time


def estimate_duration(drivers, rebuild_dataset):

    calculate_driver_results = partial(classification_model.calculate_driver, rebuild_dataset=rebuild_dataset)

    start_time = time.time()
    timing_sample = drivers[:multiprocessing.cpu_count() * 3]
    with Pool() as test_p:
        test_p.map(calculate_driver_results, timing_sample)
    avg_time = (time.time() - start_time) / len(timing_sample)
    estimated_total_time = avg_time * len(drivers)
    estimated_completion_time = start_time + estimated_total_time
    print("\n+-----------------+")
    print("Average time per driver {0:.1f}".format(avg_time))
    print("Estimated completion time: {0}".format(time.ctime(estimated_completion_time)))
    print("+-----------------+\n")


def write_results_to_file(probability_results, test):
    for item in probability_results:
        for row in item:
            file_handling.write_to_submission_file("{0}_{1},{2:.6f}".format(
                int(row[0]), int(row[1]), row[3]), test=test)


def aggregate_model_results(models, aggregate):
    for model in models.values():
        if model.name in aggregate.keys():
            aggregate[model.name].scores.extend(model.scores)
            aggregate[model.name].count += model.count
        else:
            aggregate[model.name] = model


def summarize_model_statistics(results, test=False):
    print("\nModel statistics: +---------------------+")
    header = "{0:<50}{1:>12}{2:>12}{3:>9}".format("Name", "AUC", "Variance", "Count")
    print(header)
    file_handling.write_to_model_stats_file(header, overwrite=True, test=test)

    for model in results.values():
        line = "{0:<50}{1:>12}{2:>12}{3:>9}".format(model.name,
                                                    '{0:.3f}'.format(model.get_score()),
                                                    '{0:.6f}'.format(model.get_variance()),
                                                    model.count)
        print(line)
        file_handling.write_to_model_stats_file(line, overwrite=False, test=test)


def main(test=False, rebuild_dataset=False):
    print("This is main()")
    start_time = time.time()
    drivers = file_handling.get_drivers()
    file_handling.write_to_submission_file("driver_trip,prob", overwrite=True, test=test)
    if test:
        drivers = drivers[:24]
    else:
        estimate_duration(drivers, rebuild_dataset)

    calculate_driver_results = partial(classification_model.calculate_driver, rebuild_dataset=rebuild_dataset)

    with Pool() as p:
        results = p.map(calculate_driver_results, drivers)

    probability_results = []
    aggregate_results = {}
    for result, models in results:
        probability_results.append(result)
        aggregate_model_results(models, aggregate_results)

    write_results_to_file(probability_results, test)

    summarize_model_statistics(aggregate_results, test)

    total_time = time.time()-start_time
    print("\n+-----------------+")
    print("Total time for {0} drivers   -----> {1:.1f}  sec".format(len(drivers), total_time))
    print("Per driver                 -----> {0:.1f} sec".format(total_time/len(drivers)))
    print("+-----------------+")


if __name__ == '__main__':
    main(test=False, rebuild_dataset=False)