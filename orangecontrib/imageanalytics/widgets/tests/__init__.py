def patch_import():
    try:
        import builtins
    except ImportError:
        # In Python 2 builtins are __builtin__
        import __builtin__ as builtins
    realimport = builtins.__import__

    def import_mock(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tensorflow":
            raise ImportError
        return realimport(name, globals, locals, fromlist, level)

    builtins.__import__ = import_mock
