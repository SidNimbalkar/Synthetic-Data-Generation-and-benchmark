import numpy as np
import pandas as pd

class BaseSynthesizer:
    """Base class for synthesizer"""

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        pass

    def sample(self, samples):
        pass

    def fit_sample(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.fit(data, categorical_columns, ordinal_columns)
        return self.sample(data.shape[0])
		

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"

class Transformer:

    @staticmethod
    def get_metadata(data, categorical_columns=tuple(), ordinal_columns=tuple()):
        meta = []

        df = pd.DataFrame(data)
        for index in df:
            column = df[index]

            if index in categorical_columns:
                mapper = column.value_counts().index.tolist()
                meta.append({
                    "name": index,
                    "type": CATEGORICAL,
                    "size": len(mapper),
                    "i2s": mapper
                })
            elif index in ordinal_columns:
                value_count = list(dict(column.value_counts()).items())
                value_count = sorted(value_count, key=lambda x: -x[1])
                mapper = list(map(lambda x: x[0], value_count))
                meta.append({
                    "name": index,
                    "type": ORDINAL,
                    "size": len(mapper),
                    "i2s": mapper
                })
            else:
                meta.append({
                    "name": index,
                    "type": CONTINUOUS,
                    "min": column.min(),
                    "max": column.max(),
                })

        return meta

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data):
        raise NotImplementedError
		
class UniformSynthesizer(BaseSynthesizer):
    """UniformSynthesizer."""

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.dtype = data.dtype
        self.shape = data.shape
        self.meta = Transformer.get_metadata(data, categorical_columns, ordinal_columns)

    def sample(self, samples):
        data = np.random.uniform(0, 1, (samples, self.shape[1]))

        for i, c in enumerate(self.meta):
            if c['type'] == CONTINUOUS:
                data[:, i] = data[:, i] * (c['max'] - c['min']) + c['min']
            else:
                data[:, i] = (data[:, i] * (1 - 1e-8) * c['size']).astype('int32')

        return data.astype(self.dtype)