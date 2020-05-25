# 文件可以选择接收的内容级别(debug, info), 屏幕上打印的东西可以选择(debug, info, silent)
# 模型的输出由文件与屏幕的最低等级决定
import logging
import os
import sys


LEVEL = {"debug": logging.DEBUG, "info": logging.INFO}
# DEFAULT_FORMATTER = logging.Formatter(fmt='%(message)s', datefmt='%Y/%m/%d %I:%M:%S')
DEFAULT_FORMATTER = logging.Formatter(fmt='%(message)s')


class Logger(object):
    def __init__(self, silent=False, slevel="debug", to_disk=False, log_file=None, flevel="debug", **kwargs):
        self.silent = silent
        self.slevel = LEVEL[slevel.lower()]
        self.to_disk = to_disk
        self.log_file = log_file
        self.flevel = LEVEL[flevel.lower()]
        self.formatter = kwargs.get("formatter", DEFAULT_FORMATTER)
        self.log = logging.getLogger()
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = False
        if not self.silent:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(self.slevel)
            ch.setFormatter(self.formatter)
            self.log.addHandler(ch)
        if self.to_disk:
            dirname = os.path.split(self.log_file)[0]
            if not os.path.exists(dirname) and dirname:  # a little dirty
                os.makedirs(dirname)
            fh = logging.FileHandler(self.log_file, mode='w')
            fh.setLevel(self.flevel)
            fh.setFormatter(self.formatter)
            self.log.addHandler(fh)

    def info(self, *args, **kwargs):
        self.log.info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        self.log.debug(*args, **kwargs)
