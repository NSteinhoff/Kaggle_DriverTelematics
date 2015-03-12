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


def generate_model_frequencies(models, test=False, plot=False):
    model_counts = {}
    for model in models:
        name = model
        if name in model_counts.keys():
            model_counts[name] += 1
        else:
            model_counts[name] = 1

    file_handling.write_to_model_frequencies_file("model,frequency", overwrite=True, test=test)
    for model, count in model_counts.items():
        line = "{0},{1}".format(model, count)
        file_handling.write_to_model_frequencies_file(line, test=test)

    if plot:
        plot_model_counts(model_counts)
    return model_counts


def plot_model_counts(model_counts):
    models = []
    counts = []
    for model, count in model_counts.items():
        models.append(model)
        counts.append(count)

    index = np.arange(1, len(models)+1)
    width = .35

    plt.bar(index, counts, width)
    plt.ylabel('Model counts')
    plt.title('Number of time a model was chosen')
    plt.xticks(index+width/2., models)
    plt.yticks(np.arange(0, max(counts), max(counts)/5))

    plt.show()


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

    models = []
    probability_results = []
    for result, model in results:
        models.append(model)
        probability_results.append(result)

    write_results_to_file(probability_results, test)

    model_counts = generate_model_frequencies(models, test)
    print("\nModel frequencies: +---------------------+")
    for model, count in model_counts.items():
        print("{0:<30}    {1:>5}".format(model+':', count))

    total_time = time.time()-start_time
    print("\n+-----------------+")
    print("Total time for {0} drivers   -----> {1:.1f}  sec".format(len(drivers), total_time))
    print("Per driver                 -----> {0:.1f} sec".format(total_time/len(drivers)))
    print("+-----------------+")


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=2)
    main(test=False)