__author__ = 'nikosteinhoff'

import multiprocessing
from multiprocessing import Queue
from multiprocessing import Process


class ProcessManager:
    def __init__(self, function, function_args=()):
            self.function = function
            self.function_args = function_args
            self.processes = []
            self.in_queue = Queue()
            self.out_queue = Queue()
            self.cores = multiprocessing.cpu_count()
            print("\n{0} CPU cores available ---> will spawn {1} child processes.\n".format(self.cores, self.cores-1))

    def start_processing(self):
        process_args = (self.function, self.in_queue, self.out_queue)

        full_args = process_args + self.function_args

        for cpu in range(self.cores - 1):
            self.processes.append(Process(target=self.run_function, args=full_args))

        for process in self.processes:
            process.start()

        for process in self.processes:
            process.join()
            

    def run_function(self, function, in_queue, out_queue, *args):

        while not in_queue.empty():
            next_item = in_queue.get()

            return_value = function(next_item, *args)

            out_queue.put(return_value)

        return True

def test_manager():
    values = range(10)

    manager = ProcessManager(dummy_function, (10, 100))

    for item in values:
        manager.in_queue.put(item)

    manager.start_processing()

    while not manager.out_queue.empty():
        print(manager.out_queue.get())


def dummy_function(x, y, z):
    result = x * y + z
    return result


if __name__ == '__main__':
    print("Running as main")
    test_manager()