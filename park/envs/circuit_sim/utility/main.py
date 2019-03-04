import abc
import argparse
import contextlib
import datetime
import importlib
import os
import shutil

import git

from circuit import RemoteContext, LocalContext, Context
from utility.learn.tf import set_cuda_devices, set_tflog_level
from utility.helper import format_table
from utility.io import dump_json, load_pickle, dump_pickle
from utility.logging import StructuredFormatterBuilder, get_logfile_handler, get_logger, get_console_handler
from utility.misc import nested_update, AttrDict

__all__ = ['Main', 'TFMain']


class Main(object, metaclass=abc.ABCMeta):
    def __init__(self):
        self.args = None
        self.eval_kwargs = None
        self.algo_kwargs = None
        self.version = None

        self.history_time = None
        self.current_time = None

        self.logger_path = None
        self.temp_path = None
        self.save_path = None
        self.board_path = None

        self.logger = None
        self.evaluator = None

    @staticmethod
    def get_version(ignore_uncommit, ignore_unstage, ignore_untrack):
        repo = git.Repo()
        uncommit = list(repo.index.diff('HEAD'))
        unstage = list(repo.index.diff(None))
        untrack = bool(repo.untracked_files)
        if not ignore_untrack:
            assert not untrack, 'The repository has untracked files.'
        if not ignore_unstage:
            assert not unstage, 'The repository has unstaged files.'
        if not ignore_uncommit:
            assert not uncommit, 'The repository has uncommitted files.'
        unstable = bool(uncommit or unstage or untrack)
        version = repo.head.object.hexsha
        version = repo.git.rev_parse(version, short=8)
        return '@' + version if unstable else version

    @staticmethod
    def copy_unstable_files(target, ignore_paths):
        uncommit_target_path = os.path.join(target, 'uncommit')
        unstage_target_path = os.path.join(target, 'unstage')
        untrack_target_path = os.path.join(target, 'untrack')

        os.makedirs(uncommit_target_path)
        os.makedirs(unstage_target_path)
        os.makedirs(untrack_target_path)

        repo = git.Repo()

        for i, diff in enumerate(repo.index.diff('HEAD', create_patch=True, binary=True, R=True)):
            with open(os.path.join(uncommit_target_path, f'patch_{i}'), 'w') as writer:
                writer.write(str(diff))

        for i, diff in enumerate(repo.index.diff(None, create_patch=True, binary=True)):
            with open(os.path.join(unstage_target_path, f'patch_{i}'), 'w') as writer:
                writer.write(str(diff))

        untrack_mapping = []
        counter = 0
        ignore_paths = [os.path.realpath(path) for path in ignore_paths]
        for path in repo.untracked_files:
            path = os.path.realpath(path)
            if any(path.startswith(ignore_path) for ignore_path in ignore_paths):
                continue
            counter += 1
            shutil.copy(path, os.path.join(untrack_target_path, str(counter)))
            untrack_mapping.append(path)

        dump_json(untrack_mapping, os.path.join(target, 'untrack-mapping.json'))

    @classmethod
    def load_kwargs(cls, path):
        algo_kwargs, eval_kwargs = {}, {}

        def update_kwargs(data):
            if hasattr(data, 'algo_kwargs'):
                nested_update(algo_kwargs, data.algo_kwargs)
            if hasattr(data, 'eval_kwargs'):
                nested_update(eval_kwargs, data.eval_kwargs)

        prefix = ''
        for name in path[:-1]:
            prefix += name + '.'
            update_kwargs(importlib.import_module(prefix + 'defaults'))
        update_kwargs(importlib.import_module(prefix + path[-1]))
        return AttrDict.nested_attr(algo_kwargs), AttrDict.nested_attr(eval_kwargs)

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('kwargs')
        parser.add_argument('--remote-host', type=str, default=None)
        parser.add_argument('--remote-port', type=int, default=None)
        parser.add_argument('--run-path', type=str, default='./records')
        parser.add_argument('--save-steps', type=int, default=0)
        parser.add_argument('--restore-time', type=str, default=None)
        parser.add_argument('--restore-step', type=int, default=None)
        parser.add_argument('--steps', type=int, default=300)
        parser.add_argument('--debug', type=str, default='onerror')
        parser.add_argument('--ignore-uncommit', action='store_true')
        parser.add_argument('--ignore-unstage', action='store_true')
        parser.add_argument('--ignore-untrack', action='store_true')
        parser.add_argument('--ignore-changes', action='store_true')
        return parser

    def parse_args(self):
        self.args = self.get_parser().parse_args()
        self.args.kwargs = self.args.kwargs.replace('/', '.').replace('.py', '')
        self.algo_kwargs, self.eval_kwargs = self.load_kwargs(self.args.kwargs.split('.'))

    def setup_time(self):
        self.current_time = str(datetime.datetime.now())
        if self.args.restore_time is not None and self.args.restore_step is not None:
            self.history_time = self.args.restore_time
        else:
            self.history_time = self.current_time

    def setup_path(self):
        self.version = self.get_version(self.args.ignore_uncommit, self.args.ignore_unstage, self.args.ignore_untrack)

        def partial_path(name):
            tags = [self.args.kwargs, self.version]
            return os.path.join(self.args.run_path, name, *tags)

        if self.version.startswith('@') and not self.args.ignore_changes:
            self.copy_unstable_files(os.path.join(partial_path('git'), self.current_time), [self.args.run_path])

        self.temp_path = os.path.join(partial_path('temp'), self.history_time)
        self.save_path = os.path.join(partial_path('save'), self.history_time)
        self.board_path = os.path.join(partial_path('board'), self.history_time)
        self.logger_path = os.path.join(partial_path('logger'), self.current_time + '.log')

    def setup_logger(self):
        os.makedirs(os.path.dirname(self.logger_path), exist_ok=True)
        formatter = StructuredFormatterBuilder(use_time=False, use_name=False, use_level=False).get_formatter()
        logfile_handler = get_logfile_handler(self.logger_path, formatter=formatter)
        console_handler = get_console_handler(formatter=formatter)
        self.logger = get_logger(self.__class__.__name__, logfile_handler, console_handler, propagate=False)
        self.logger.setLevel('INFO')

    @contextlib.contextmanager
    def open_evaluator(self):
        host = self.args.remote_host
        port = self.args.remote_port
        assert self.args.debug in ('none', 'always', 'onerror')
        if host is not None and port is not None:
            context = RemoteContext(host, port, debug=self.args.debug)
        else:
            context = LocalContext(self.temp_path, debug=self.args.debug)

        with context:
            self.evaluator = self.eval_kwargs.circuit().evaluator()
            for k in self.evaluator.parameters:
                if k in self.eval_kwargs.lower_bound and k in self.eval_kwargs.upper_bound:
                    self.evaluator.set_bound(k, self.eval_kwargs.lower_bound[k], self.eval_kwargs.upper_bound[k])
                else:
                    assert k not in self.eval_kwargs.lower_bound and k not in self.eval_kwargs.upper_bound
                    self.evaluator.preset(k, self.eval_kwargs.preset[k])
            yield
            self.evaluator = None

    @abc.abstractmethod
    def mainloop(self, from_step, steps):
        pass

    def load_model(self, model, load_step):
        self.logger.info(f'==> Loading model from step {load_step} at {self.history_time}...')
        sav_path = os.path.join(self.save_path, str(load_step))
        model.load(sav_path)
        result = load_pickle(os.path.join(sav_path, 'args.pkl'))
        self.logger.info(f'    Model loaded from "{sav_path}"\n')
        return result

    def save_model(self, model, save_step, *args):
        self.logger.info(f'==> Saving model from step {save_step} at {self.history_time}...')
        sav_path = os.path.join(self.save_path, str(save_step))
        if os.path.exists(sav_path):
            shutil.rmtree(sav_path)
        os.makedirs(sav_path)
        model.save(sav_path)
        dump_pickle(args, os.path.join(sav_path, 'args.pkl'), protocol=4)
        self.logger.info(f'    Model saved in "{sav_path}"\n')
        return sav_path

    def is_save_step(self, step):
        return self.args.save_steps > 0 and step % self.args.save_steps == 0

    def get_misc_config(self):
        return [
            ('Repo version', self.version),
            ('History time', self.history_time),
            ('Current time', self.current_time),
            ('Save path', self.save_path),
            ('Temp path', self.temp_path),
            ('Board path', self.board_path),
            ('Logger path', self.logger_path)
        ]

    def get_eval_config(self):
        return [
            ('Name', self.evaluator.circuit.__class__.__name__),
            ('Context', Context.current_context()),
            ('Lower bound', self.evaluator.lower_bound),
            ('Upper bound', self.evaluator.upper_bound),
            ('Benchmark', self.eval_kwargs.benchmark)
        ]

    @staticmethod
    def format_config(config):
        if isinstance(config, dict):
            config = format_table(config.keys(), config.values())
        else:
            config = format_table(*zip(*config))
        return '\n'.join(config) + '\n'

    def log_misc_config(self):
        self.logger.info(f'==> {self.__class__.__name__} on Circuit {self.evaluator.circuit.__class__.__name__}')
        self.logger.info(self.format_config(self.get_misc_config()))

    def log_eval_config(self):
        self.logger.info(f'==> Circuit Information')
        self.logger.info(self.format_config(self.get_eval_config()))

    def log_algo_config(self):
        self.logger.info(f'==> Algorithm Information')
        self.logger.info(self.format_config(self.algo_kwargs))

    def log_config(self):
        self.log_misc_config()
        self.log_eval_config()
        self.log_algo_config()

    def setup(self):
        self.setup_time()
        self.setup_path()
        self.setup_logger()

    def run(self):
        self.parse_args()
        self.setup()

        with self.open_evaluator():
            self.log_config()

            try:
                if self.args.restore_step is not None and self.args.restore_time is not None:
                    self.mainloop(self.args.restore_step, self.args.steps)
                else:
                    self.mainloop(0, self.args.steps)
            except:
                self.logger.exception('!!! ==> Exception occurred')


class TFMain(Main, metaclass=abc.ABCMeta):
    def setup(self):
        super(TFMain, self).setup()
        set_tflog_level(self.args.tflog_level)
        if self.args.gpu:
            self.args.gpu = self.args.gpu.split(',')
            set_cuda_devices(self.args.gpu)

    def get_parser(self):
        parser = super().get_parser()
        parser.add_argument('--tflog-level', type=int, default=3)
        parser.add_argument('--gpu', type=str, default=None)
        return parser

    def get_misc_config(self):
        config = super().get_misc_config()
        config.append(('TFLog Level', self.args.tflog_level))
        config.append(('GPUs', self.args.gpu if self.args.gpu is not None else 'On all GPUs'))
        return config
