import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import KBinsDiscretizer
import json
from pomegranate import BayesianNetwork, ConditionalProbabilityTable, DiscreteDistribution


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


class DiscretizeTransformer(Transformer):
    """Discretize continuous columns into several bins.
    Attributes:
        meta
        column_index
        discretizer(sklearn.preprocessing.KBinsDiscretizer)
    Transformation result is a int array.
    """

    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.meta = None
        self.column_index = None
        self.discretizer = None

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.meta = self.get_metadata(data, categorical_columns, ordinal_columns)
        self.column_index = [
            index for index, info in enumerate(self.meta) if info['type'] == CONTINUOUS]

        self.discretizer = KBinsDiscretizer(
            n_bins=self.n_bins, encode='ordinal', strategy='uniform')

        if not self.column_index:
            return

        self.discretizer.fit(data[:, self.column_index])

    def transform(self, data):
        """Transform data discretizing continous values.
        Args:
            data(pandas.DataFrame)
        Returns:
            numpy.ndarray
        """
        if self.column_index == []:
            return data.astype('int')

        data[:, self.column_index] = self.discretizer.transform(data[:, self.column_index])
        return data.astype('int')

    def inverse_transform(self, data):
        if self.column_index == []:
            return data

        data = data.astype('float32')
        data[:, self.column_index] = self.discretizer.inverse_transform(data[:, self.column_index])
        return data


class CLBNSynthesizer(BaseSynthesizer):
    """CLBNSynthesizer."""

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.discretizer = DiscretizeTransformer(n_bins=15)
        self.discretizer.fit(data, categorical_columns, ordinal_columns)
        discretized_data = self.discretizer.transform(data)
        self.model = BayesianNetwork.from_samples(discretized_data, algorithm='chow-liu')

    def bn_sample(self, num_samples):
        """Sample from the bayesian network.
        Args:
            num_samples(int): Number of samples to generate.
        """
        nodes_parents = self.model.structure
        processing_order = []

        while len(processing_order) != len(nodes_parents):
            update = False

            for id_, parents in enumerate(nodes_parents):
                if id_ in processing_order:
                    continue

                flag = True
                for parent in parents:
                    if parent not in processing_order:
                        flag = False

                if flag:
                    processing_order.append(id_)
                    update = True

            assert update

        data = np.zeros((num_samples, len(nodes_parents)), dtype='int32')
        for current in processing_order:
            distribution = self.model.states[current].distribution
            if isinstance(distribution, DiscreteDistribution):
                data[:, current] = distribution.sample(num_samples)
            else:
                assert isinstance(distribution, ConditionalProbabilityTable)
                output_size = list(distribution.keys())
                output_size = max([int(x) for x in output_size]) + 1

                distribution = json.loads(distribution.to_json())
                distribution = distribution['table']

                distribution_dict = {}

                for row in distribution:
                    key = tuple(np.asarray(row[:-2], dtype='int'))
                    output = int(row[-2])
                    p = float(row[-1])

                    if key not in distribution_dict:
                        distribution_dict[key] = np.zeros(output_size)

                    distribution_dict[key][int(output)] = p

                parents = nodes_parents[current]
                conds = data[:, parents]
                for _id, cond in enumerate(conds):
                    data[_id, current] = np.random.choice(
                        np.arange(output_size),
                        p=distribution_dict[tuple(cond)]
                    )

        return data

    def sample(self, num_samples):
        data = self.bn_sample(num_samples)
        return self.discretizer.inverse_transform(data)