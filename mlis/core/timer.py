import time
from contextlib import contextmanager

class Timer():
    def __init__(self, time_limit, time_mult):
        self.time_limit = time_limit
        self.time_mult = time_mult
        self.start_time = time.time()
        self.pause_time = 0.0

    @contextmanager
    def pause(self):
        pause_start = time.time()
        yield None
        pause_end = time.time()
        self.pause_time += pause_end-pause_start

    def get_time_left(self):
        return self.time_limit - self.get_execution_time()

    def get_execution_time(self):
        return (time.time() - self.start_time - self.pause_time) * self.time_mult

    def get_pause_time(self):
        return self.pause_time
