from __future__ import print_function
import sys
import numpy


class Wrapper(object):
    def __init__(self, wrapped_module, end_module):
        self.wrapped_module = wrapped_module
        self.end_module = end_module
        self.attribute_names = []

    def __getattribute__(self, item):
        return object.__getattribute__(self,item)

    def __members__(self):
        return object.__members__(self)

    def __methods__(self):
        return object.__methods__(self)

    def trait_names(self):
        return object.trait_names(self)

    def _getAttributeNames(self):
        return object._getAttributeNames(self)

    def __getattr__(self, attr):
        try:
            orig_attr = self.wrapped_module.__getattribute__(attr)
        except (AttributeError,TypeError) as e:
            self.eprint("The following function doesn't exist %s" % e)
        else:
            if callable(orig_attr):
                def hooked(*args, **kwargs):
                    new_args = self.pre(*args, **kwargs)
                    result = orig_attr(*new_args, **kwargs)
                    return self.post(result)
                return hooked
            else:
                try:
                    from types import ModuleType
                    import sys
                    if type(orig_attr) is ModuleType:
                        orig = orig_attr.__name__
                        if self.wrapped_module.__name__ in orig:
                            dest = orig_attr.__name__.replace(self.wrapped_module.__name__,self.end_module)
                            subModule = sys.modules[orig_attr.__name__]
                            wrap = type(self)(subModule, dest)
                            sys.modules[dest] = wrap
                            return wrap
                    return orig_attr
                except (AttributeError,TypeError) as e:
                    print("%s error creating wrapper to module %s" % (e, orig_attr.__name__))
                    return self.handleError()
        return getattr(self,"handleError")

    def eprint(*args, **kwargs):
        import sys
        print(*args, file=sys.stderr, **kwargs)

    def pre(self,*args,**kwargs):
        self.eprint("Calling NumPy function with the following parameters:")
        new_args = []
        for i in args:
            self.eprint("%s of type %s" % (str(i),str(type(i))))
            if(callable(i)):
                self.eprint("metode")
            elif("matlib.core.numeric.array" in str(i.__class__)):
                new_args.append(i.__numpy_array__)
                continue
            new_args.append(i)
        return new_args

        print(kwargs)
        for key in kwargs:
            self.eprint("%s of type %s" % (str(kwargs[key]), str(type(kwargs[key]))))

    def convert(self,result):
        print("Trying to convert")
        if "numpy.ndarray" in str(result.__class__):
            from matlib.core.numeric import array
            newArray = array(result.tolist())
            return newArray

        is_tuple = False
        print("[POST] Entering in Post " + str(result))
        if isinstance(result, tuple):
            result = list(result)
            is_tuple = True
        try:
            iterator = iter(result)
            pairs = list(enumerate(result))
            for p in pairs:
                result[p[0]] = self.convert(p[1])
        except TypeError:
            return result
        except:
            import sys
            self.eprint("Unexpected error:", sys.exc_info()[0])
        if is_tuple:
            result = tuple(result)

        return result

    def post(self, result):
        return self.convert(result)

    def handleError(self):
        print("Error handler")

sys.modules[__name__] = Wrapper(numpy, __name__)
