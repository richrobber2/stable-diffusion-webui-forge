import contextlib
import importlib


always_true_func = lambda *args, **kwargs: True


class CondFunc:
    def __new__(cls, orig_func, sub_func, cond_func=always_true_func):
        self = super(CondFunc, cls).__new__(cls)
        if isinstance(orig_func, str):
            func_path = orig_func.split('.')
            for i in range(len(func_path)-1, -1, -1):
                with contextlib.suppress(ImportError):
                    resolved_obj = importlib.import_module('.'.join(func_path[:i]))
                    break
            try:
                for attr_name in func_path[i:-1]:
                    resolved_obj = getattr(resolved_obj, attr_name)
                orig_func = getattr(resolved_obj, func_path[-1])
                setattr(resolved_obj, func_path[-1], lambda *args, **kwargs: self(*args, **kwargs))
            except AttributeError:
                print(f"Warning: Failed to resolve {orig_func} for CondFunc hijack")
        self.__init__(orig_func, sub_func, cond_func)
        return lambda *args, **kwargs: self(*args, **kwargs)
    def __init__(self, orig_func, sub_func, cond_func):
        self.__orig_func = orig_func
        self.__sub_func = sub_func
        self.__cond_func = cond_func
    def __call__(self, *args, **kwargs):
        if not self.__cond_func or self.__cond_func(self.__orig_func, *args, **kwargs):
            return self.__sub_func(self.__orig_func, *args, **kwargs)
        else:
            return self.__orig_func(*args, **kwargs)
