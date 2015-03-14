__author__ = "nikosteinhoff"
__email__ = "niko.steinhoff@gmail.com"

import numpy as np
from src import file_handling
from src import classification_model
import multiprocessing
from multiprocessing import Pool
import matplotlib.pyplot as plt

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


def write_results_to_file(probability_results, test):
    for item in probability_results:
        for row in item:
            file_handling.write_to_submission_file("{0}_{1},{2:.6f}".format(
                int(row[0]), int(row[1]), row[3]), test=test)


def calculate_feature_importances(data):
    all_importances = {}
    for dict in data:
        for name, value in dict.items():
            if name in all_importances.keys():
                all_importances[name].append(value)
            else:
                all_importances[name] = [value]

    avg_importances = {}
    for name, values in all_importances.items():
        avg_importances[name] = np.array(values).mean()

    return avg_importances


def aggregate_model_results(models, aggregate):
    for model in models.values():
        if model.name in aggregate.keys():
            aggregate[model.name].scores.extend(model.scores)
            aggregate[model.name].count += model.count
        else:
            aggregate[model.name] = model


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

    probability_results = []
    feature_importances = []
    aggregate_results = {}
    for result, models, feature_importance in results:
        probability_results.append(result)
        aggregate_model_results(models, aggregate_results)
        feature_importances.append(feature_importance)

    write_results_to_file(probability_results, test)

    print("\nModel statistics: +---------------------+")
    print("{0:<30}{1:>12}{2:>12}{3:>9}".format("Name", "AUC", "Variance", "Count"))
    for model in aggregate_results.values():
        print("{0:<30}{1:>12}{2:>12}{3:>9}".format(model.name,
                                                    '{0:.3f}'.format(model.get_score()),
                                                    '{0:.6f}'.format(model.get_variance()),
                                                    model.count))

    overall_feature_importances = calculate_feature_importances(feature_importances)
    print("\nFeature importances: +---------------------+")
    for feature, importance in overall_feature_importances.items():
        print("{0:<30}{1:>15}".format(feature+':', '{0:.4f}'.format(importance)))

    total_time = time.time()-start_time
    print("\n+-----------------+")
    print("Total time for {0} drivers   -----> {1:.1f}  sec".format(len(drivers), total_time))
    print("Per driver                 -----> {0:.1f} sec".format(total_time/len(drivers)))
    print("+-----------------+")


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=2)
    main(test=False)