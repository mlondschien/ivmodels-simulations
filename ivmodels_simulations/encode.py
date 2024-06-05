import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    # https://stackoverflow.com/a/47626762
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
