import os

def mkdir_save(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)

def merge_dict(dict1, dict2):
    res = {**dict1, **dict2}
    return res