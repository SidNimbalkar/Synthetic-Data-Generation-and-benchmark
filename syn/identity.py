import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import KBinsDiscretizer

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

class IdentitySynthesizer(BaseSynthesizer):
    """Trivial synthesizer.
    Returns the same exact data that is used to fit it.
    """

    def fit(self, train_data, *args):
        self.data = pd.DataFrame(train_data)

    def sample(self, samples):
        return self.data.sample(samples, replace=True).values