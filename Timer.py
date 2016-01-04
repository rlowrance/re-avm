import time


class Timer(object):
    def __init__(self):
        self.start_clock = time.clock()  # processor time in seconds
        self.start_time = time.time()  # time in seconds since the epoch

        pass

    def elapsed_cpu_seconds(self):
        return time.clock() - self.start_clock

    def elapsed_wallclock_seconds(self):
        return time.time() - self.start_time
