import os
import wget
import zipfile
import random
import numpy as np
import pandas as pd
import park


def load_traces(trace, cache_size):
    if trace == 'test':
        trace_folder = park.__path__[0] + '/envs/cache/traces/'

        if not os.path.exists(trace_folder):
            os.mkdir(trace_folder)
            wget.download(
                'https://www.dropbox.com/s/bfed1jk38sfvpez/test_trace.zip?dl=1',
                out=trace_folder)
            with zipfile.ZipFile(
                park.__path__[0] + '/envs/cache/traces/test_trace.zip', 'r') as zip_f:
                zip_f.extractall(park.__path__[0] + '/envs/cache/traces/test_trace/')            

        rnd = random.randint(1, 1001)
        print('Load #%i trace for cache size of %i' % (rnd, cache_size))

        # load time, request id, request size
        df = pd.read_csv(trace_folder + 'test_trace/test_' + str(rnd) + '.tr', sep=' ', header=None)
        # remaining cache size, object last access time
        df[3], df[4] = cache_size, 0
        
    else:
        # load user's trace
        df = pd.read_csv(trace, sep=' ', header=None)
        df[3], df[4] = cache_size, 0
    
    return df
