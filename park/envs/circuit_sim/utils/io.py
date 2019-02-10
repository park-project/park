import contextlib
import os
import shutil

import uuid

import datetime

try:
    import cPickle as pickle
except ImportError:
    import pickle

__all__ = ['dumps_pickle', 'dump_pickle', 'load_pickle', 'loads_pickle', 'open_tmp_path']


def dump_pickle(data, path, **kwargs):
    with open(path, 'wb') as writer:
        pickle.dump(data, writer, **kwargs)


def dumps_pickle(data, **kwargs):
    return pickle.dumps(data, **kwargs)


def load_pickle(path):
    with open(path, 'rb') as reader:
        return pickle.load(reader)


def loads_pickle(data):
    return pickle.loads(data)


@contextlib.contextmanager
def open_tmp_path(base_path, tmp_type='time', keep_on_error=False):
    if tmp_type == 'time':
        tmp_name = str(datetime.datetime.now())
    elif tmp_type == 'timepid':
        tmp_name = str(datetime.datetime.now()) + '-' + str(os.getpid())
    elif tmp_type == 'uuid':
        tmp_name = str(uuid.uuid4())
    else:
        raise ValueError(f'Cannot use temporary type of "{tmp_type}"')
    tmp_path = os.path.join(base_path, tmp_name or str(uuid.uuid4()))
    os.makedirs(tmp_path, exist_ok=True)
    try:
        yield tmp_path
    except Exception:
        if not keep_on_error:
            shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    shutil.rmtree(tmp_path, ignore_errors=True)
