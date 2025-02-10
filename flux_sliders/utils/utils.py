# -*- encoding: utf-8 -*-
from datetime import datetime
from pathlib import Path


def mkdir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_cur_timestamp():
    return datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")
