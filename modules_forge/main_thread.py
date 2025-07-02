# This file is the main thread that handles all gradio calls for major t2i or i2i processing.
# Other gradio calls (like those from extensions) are not influenced.
# By using one single thread to process all major calls, model moving is significantly faster.


import time
import traceback
import threading


lock = threading.Lock()
condition = threading.Condition(lock)
last_id = 0
waiting_list = []
finished_list = []
last_exception = None


class Task:
    def __init__(self, task_id, func, args, kwargs):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.exception = None
        self.done_event = threading.Event()

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
            self.done_event.set()


def loop():
    global lock, last_id, waiting_list, finished_list
    while True:
        with condition:
            while not waiting_list:
                condition.wait()
            task = waiting_list.pop(0)
        task.work()
        with lock:
            finished_list.append(task)


def async_run(func, *args, **kwargs):
    global lock, last_id, waiting_list, finished_list
    with condition:
        last_id += 1
        new_task = Task(task_id=last_id, func=func, args=args, kwargs=kwargs)
        waiting_list.append(new_task)
        condition.notify()
    return new_task.task_id


def run_and_wait_result(func, *args, **kwargs):
    global lock, last_id, waiting_list, finished_list
    with condition:
        last_id += 1
        new_task = Task(task_id=last_id, func=func, args=args, kwargs=kwargs)
        waiting_list.append(new_task)
        condition.notify()
    new_task.done_event.wait()
    if new_task.exception is not None:
        raise new_task.exception
    with lock:
        if new_task in finished_list:
            finished_list.remove(new_task)
    return new_task.result

