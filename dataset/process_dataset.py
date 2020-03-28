import pandas as pd
import numpy as np
import json

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"	

def get_data(path):
    return pd.read_csv(str(path))

def get_columns(df):
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    ord_cols = []
    return (num_cols.tolist(),cat_cols,ord_cols,cols)

def get_column_type_list(continuous_columns,categorical_columns,ordinal_columns,columns,df):
    col_type = []
    for i in range(len(continuous_columns)):
        col_type.append((continuous_columns[i], CONTINUOUS))
    for i in range(len(categorical_columns)):
        col_type.append((categorical_columns[i], CATEGORICAL))
    for i in range(len(ordinal_columns)):
        col_type.append((ordinal_columns[i], ORDINAL, df[ordinal_columns[i]].unique().tolist()))

    col_type = [tuple for x in columns for tuple in col_type if tuple[0] == x]
    return(col_type)
	
def get_metadata(column_type,df):
    meta = []
    for id_, info in enumerate(column_type):
        if info[1] == CONTINUOUS:
            meta.append({
                "name": info[0],
                "type": info[1],
                "min": np.min(df.iloc[:, id_].values.astype('float')),
                "max": np.max(df.iloc[:, id_].values.astype('float'))
            })
        else:
            if info[1] == CATEGORICAL:
                value_count = list(dict(df.iloc[:, id_].value_counts()).items())
                value_count = sorted(value_count, key=lambda x: -x[1])
                mapper = list(map(lambda x: x[0], value_count))
            else:
                mapper = info[2]

            meta.append({
                "name": info[0],
                "type": info[1],
                "size": len(mapper),
                "i2s": mapper
            })
    return meta
	
def project_table(data, meta):
    values = np.zeros(shape=data.shape, dtype='float32')

    for id_, info in enumerate(meta):
        if info['type'] == CONTINUOUS:
            values[:, id_] = data.iloc[:, id_].values.astype('float32')
        else:
            mapper = dict([(item, id) for id, item in enumerate(info['i2s'])])
            mapped = data.iloc[:, id_].apply(lambda x: mapper[x]).values
            values[:, id_] = mapped
            mapped = data.iloc[:, id_].apply(lambda x: mapper[x]).values
    return values
	
def prep_npz_file(tdata,file_name,df):
    np.random.seed(0)
    np.random.shuffle(tdata)
    t_train = tdata[:-int(df.shape[0]/2)]
    t_test = tdata[-int(df.shape[0]/2):]
    np.savez(file_name+".npz", train=t_train, test=t_test)
	
def prep_meta_file(meta,problem_type,filename):
    metadata = {}
    metadata = {'columns': meta, 'problem_type': problem_type}
    with open(filename+".json", 'w') as f:
        json.dump(metadata, f, sort_keys=True, indent=4, separators=(',', ': '))