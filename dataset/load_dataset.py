import pandas as pd
import numpy as np
import json

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"	

def _get_columns(metadata):
    categorical_columns = list()
    ordinal_columns = list()
    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)
        elif column['type'] == ORDINAL:
            ordinal_columns.append(column_idx)

    return categorical_columns, ordinal_columns
	
def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)

def _load_file(filename, loader):
    local_path = filename
    return loader(local_path)

def load_dataset(name, benchmark=False):
    data = _load_file(name + '.npz', np.load)
    meta = _load_file(name + '.json', _load_json)

    categorical_columns, ordinal_columns = _get_columns(meta)

    train = data['train']
    if benchmark:
        return train, data['test'], meta, categorical_columns, ordinal_columns

    return train, categorical_columns, ordinal_columns