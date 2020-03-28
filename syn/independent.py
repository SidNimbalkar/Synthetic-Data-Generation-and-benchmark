import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

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


class IndependentSynthesizer(BaseSynthesizer):
    """docstring for IdentitySynthesizer."""

    def __init__(self, gmm_n=5):
        self.gmm_n = gmm_n

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.dtype = data.dtype
        self.meta = Transformer.get_metadata(data, categorical_columns, ordinal_columns)

        self.models = []
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                model = GaussianMixture(self.gmm_n)
                model.fit(data[:, [id_]])
                self.models.append(model)
            else:
                nomial = np.bincount(data[:, id_].astype('int'), minlength=info['size'])
                nomial = nomial / np.sum(nomial)
                self.models.append(nomial)

    def sample(self, samples):
        data = np.zeros([samples, len(self.meta)], self.dtype)

        for i, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                x, _ = self.models[i].sample(samples)
                np.random.shuffle(x)
                data[:, i] = x.reshape([samples])
                data[:, i] = data[:, i].clip(info['min'], info['max'])
            else:
                data[:, i] = np.random.choice(np.arange(info['size']), samples, p=self.models[i])

        return data