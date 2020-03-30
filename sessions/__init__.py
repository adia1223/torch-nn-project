import json
import os
import pickle
from copy import deepcopy
from datetime import datetime

from history.index import save_record

os.environ['TORCH_HOME'] = "D:\\torch_home"

SESSIONS_ROOT = "D:\\petrtsv\\projects\ds\\pytorch-sessions"

SESSION_FOLDER_PATTERN = "%s_%s"
SESSION_CREATION_TIME_PATTERN = "%f-%S-%M-%H-%d-%m-%Y"

CHECKPOINT_DIR = "checkpoint"
OUTPUT_DIR = "output"


def is_jsonable(data):
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(is_jsonable(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and is_jsonable(v) for k, v in data.items())
    return False


class Session(object):

    def __create__(self, name, comment, **kwargs):
        if name is None:
            raise ValueError("Session name can not be None")
        self.screen_name = name
        now = datetime.now()
        self.full_name = SESSION_FOLDER_PATTERN % (self.screen_name, now.strftime(SESSION_CREATION_TIME_PATTERN))
        self.session_dir = os.path.join(SESSIONS_ROOT, self.screen_name, self.full_name)
        self.checkpoint_dir = os.path.join(self.session_dir, CHECKPOINT_DIR)
        self.output_dir = os.path.join(self.session_dir, OUTPUT_DIR)
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.data = kwargs
        self.data['screen_name'] = self.screen_name
        self.data['full_name'] = self.full_name
        self.data['session_dir'] = self.session_dir
        self.data['checkpoint_dir'] = self.checkpoint_dir
        self.data['output_dir'] = self.output_dir
        self.data['comment'] = comment

        to_json = {}
        for key in self.data:
            if not is_jsonable(self.data[key]):
                to_json[key] = str(self.data[key])
            else:
                to_json[key] = deepcopy(self.data[key])
        with open(os.path.join(self.session_dir, self.full_name + ".json"), 'w') as fout:
            json.dump(to_json, fout, indent=4)

    def __restore__(self, data_file):
        with open(data_file, "rb") as fin:
            self.data = dict(pickle.load(fin))

    def checkpoint(self):
        with open(os.path.join(self.data['checkpoint_dir'], "session_data.pickle"), "wb") as fout:
            pickle.dump(self.data, fout)

    def build(self, name=None, comment="", state_file=None, **kwargs):
        if state_file is None:
            self.__create__(name, comment, **kwargs)
        else:
            self.__restore__(state_file)

    def save_info(self):
        with open(os.path.join(self.output_dir, "info.json"), 'w') as fout:
            json.dump(self.data, fout, indent=4)
        save_record(self.full_name, **self.data)
