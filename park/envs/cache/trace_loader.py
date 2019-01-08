import os
import wget
import zipfile
import numpy as np
import pandas as pd
import park


def load_traces(trace_name, cache_size):
    # download CDN cache traces if not existed
    trace_folder = park.__path__[0] + '/envs/cache/traces/'
    if not os.path.exists(trace_folder):
        os.mkdir(trace_folder)
    
    if not os.path.exists(trace_folder + trace_name + '.tr'):
        wget.download(
            'https://www.dropbox.com/s/qiizxqqzthnd9u7/cache_trace.zip?dl=1',
            out=trace_folder)
        with zipfile.ZipFile(
            park.__path__[0] + '/envs/cache/traces/cache_trace.zip', 'r') as zip_f:
            zip_f.extractall(park.__path__[0] + '/envs/cache/traces/')

    # load time, request id, request size
    df = pd.read_csv(trace_folder + trace_name + '.tr', sep=' ', header=None)
    # remaining cache size
    df[3] = cache_size
    # cost
    df[4] = 0
    # object last access time
    df[5] = 0
    return df
