import time
import logging
from functools import wraps
import datetime

prettyTime = lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# get current time in milliseconds
millis = lambda: int(round(time.time() * 1000))

# creating decorator to time functions
def timeit(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        time_before = millis()
        try:
            return func(*args, **kwargs)
        finally:
            time_after = millis()
            time_delta = time_after - time_before
            logging.debug(f"{func.__name__} took {time_delta // 1000} seconds {time_delta % 1000} milliseconds.")
    return _time_it
