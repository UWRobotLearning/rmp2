"""
python helper functions
"""

from contextlib import contextmanager
import time

@contextmanager
def timing(msg, verbose=True):
    if verbose:
        # print("\t---------------------------------")
        print("\t[Timer] %s started..." % (msg))
        tstart = time.time()
        yield
        print("\t[Timer] %s done in %.3f seconds" % (msg, time.time() - tstart))
        print("\t---------------------------------")
    else:
        yield

def merge_dicts(original, new_dict):
    if new_dict is not None:
        updated_dict = original.copy()
        updated_dict.update(new_dict)
    else:
        updated_dict = original
    return updated_dict
