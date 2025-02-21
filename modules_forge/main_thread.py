# This file is the main thread that handles all gradio calls for major t2i or i2i processing.
# Other gradio calls (like those from extensions) are not influenced.
# By using one single thread to process all major calls, model moving is significantly faster.


import time
import traceback
import threading


lock = threading.Lock()
last_id = 0
waiting_list = []
finished_list = []
last_exception = None

_task_condition = threading.Condition(lock)
_min_sleep = 0.001  # Minimum sleep time 1ms
_max_sleep = 0.05   # Maximum sleep time 50ms
_current_sleep = _min_sleep


class Task:
    def __init__(self, task_id, func, args, kwargs):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.exception = None
        self.event = threading.Event()  # new event to signal completion

    def work(self):
        global last_exception
        try:
            self.result = self.func(*self.args, **self.kwargs)
            self.exception = None
            last_exception = None
        except Exception as e:
            traceback.print_exc()
            print(e)
            self.exception = e
            last_exception = e
        finally:
            with lock:
                finished_list.append(self)
            self.event.set()  # signal that the task is finished


def loop():
    global lock, last_id, waiting_list, finished_list, _current_sleep
    while True:
        task = None
        with _task_condition:
            if len(waiting_list) > 0:
                task = waiting_list.pop(0)
                _current_sleep = _min_sleep  # Reset sleep time when work found
            else:
                # Wait with timeout, allows for new tasks to wake us
                _task_condition.wait(timeout=_current_sleep)
                # Exponential backoff up to max sleep time
                _current_sleep = min(_max_sleep, _current_sleep * 1.5)
                continue

        if task is not None:
            task.work()


def async_run(func, *args, **kwargs):
    global lock, last_id, waiting_list, finished_list
    with lock:
        last_id += 1
        new_task = Task(task_id=last_id, func=func, args=args, kwargs=kwargs)
        waiting_list.append(new_task)
        _task_condition.notify()  # Wake up worker thread
    return new_task.task_id


def run_and_wait_result(func, *args, **kwargs):
    # Use synchronous creation and wait for task completion
    with lock:
        global last_id
        last_id += 1
        new_task = Task(task_id=last_id, func=func, args=args, kwargs=kwargs)
        waiting_list.append(new_task)
        _task_condition.notify()  # Wake up worker thread
    new_task.event.wait()  # wait until task.work() signals completion

    # new check: if task failed, raise its exception
    if new_task.exception is not None:
        raise new_task.exception

    return new_task.result

