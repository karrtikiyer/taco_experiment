"""
Vendored from FlagOpen/TACO: https://github.com/FlagOpen/TACO/blob/main/metrics/pyext2.py

Copyright (C) 2014 Ryan Gonzalez - MIT License
RuntimeModule used for dynamic code compilation in TACO's testing framework.
"""

g_backup = globals().copy()

__version__ = '0.7'

import sys, inspect, types


def set_docstring(doc):
    def _wrap(f):
        f.__doc__ = doc
        return f
    return _wrap


if sys.version_info.major == 3:
    def modify_function(f, globals={}, name=None, code=None, defaults=None,
                        closure=None):
        if code is None: code = f.__code__
        if name is None: name = f.__name__
        if defaults is None: defaults = f.__defaults__
        if closure is None: closure = f.__closure__
        newf = types.FunctionType(code, dict(f.__globals__, **globals), name=name,
                                  argdefs=defaults, closure=closure)
        newf.__dict__.update(f.__dict__)
        return newf
    def argspec(f):
        return inspect.getfullargspec(f)
    def _exec(m, g): exec(m, g)
else:
    raise RuntimeError("Python 3+ required")


def _gettypes(args):
    return tuple(map(type, args))


class overload(object):
    _items = {}
    _types = {}

    @classmethod
    def argc(self, argc=None):
        argc = {'argc': argc}
        def _wrap(f):
            def _newf(*args, **kwargs):
                if len(args) not in self._items[f.__name__]:
                    raise TypeError("No overload of function '%s' that takes %d args" % (f.__name__, len(args)))
                return self._items[f.__name__][len(args)](*args, **kwargs)
            if f.__name__ not in self._items:
                self._items[f.__name__] = {}
            if argc['argc'] is None:
                argc['argc'] = len(argspec(f).args)
            self._items[f.__name__][argc['argc']] = f
            _newf.__name__ = f.__name__
            _newf.__doc__ = f.__doc__
            _newf.__is_overload__ = True
            _newf.__orig_arg__ = argspec(f)
            return _newf
        return _wrap


class _RuntimeModule(object):
    def __call__(self, *args, **kwargs):
        return self.from_objects(*args, **kwargs)

    @staticmethod
    @overload.argc(1)
    def from_objects(module_name_for_code_eval, **d):
        return _RuntimeModule.from_objects(module_name_for_code_eval, '', **d)

    @staticmethod
    @overload.argc(2)
    def from_objects(module_name_for_code_eval, docstring, **d):
        module = types.ModuleType(module_name_for_code_eval, docstring)
        module.__dict__.update(d)
        module.__file__ = ' '
        sys.modules[module_name_for_code_eval] = module
        return module

    @staticmethod
    @overload.argc(2)
    def from_string(module_name_for_code_eval, s):
        return _RuntimeModule.from_string(module_name_for_code_eval, '', s)

    @staticmethod
    @overload.argc(3)
    def from_string(module_name_for_code_eval, docstring, s):
        g = {}
        _exec(s, g)
        return _RuntimeModule.from_objects(module_name_for_code_eval, docstring,
                                           **dict(filter(lambda x: x[0] not in g_backup, g.items())))


RuntimeModule = _RuntimeModule()
