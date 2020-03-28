import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import KBinsDiscretizer
import torch
from torch.nn import Dropout, Linear, Module, ReLU, Sequential
from torch.nn.functional import mse_loss, softmax
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

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

class GeneralTransformer(Transformer):
    """Continuous and ordinal columns are normalized to [0, 1].
    Discrete columns are converted to a one-hot vector.
    """

    def __init__(self, act='sigmoid'):
        self.act = act
        self.meta = None
        self.output_dim = None

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.meta = self.get_metadata(data, categorical_columns, ordinal_columns)
        self.output_dim = 0
        for info in self.meta:
            if info['type'] in [CONTINUOUS, ORDINAL]:
                self.output_dim += 1
            else:
                self.output_dim += info['size']

    def transform(self, data):
        data_t = []
        self.output_info = []
        for id_, info in enumerate(self.meta):
            col = data[:, id_]
            if info['type'] == CONTINUOUS:
                col = (col - (info['min'])) / (info['max'] - info['min'])
                if self.act == 'tanh':
                    col = col * 2 - 1
                data_t.append(col.reshape([-1, 1]))
                self.output_info.append((1, self.act))

            elif info['type'] == ORDINAL:
                col = col / info['size']
                if self.act == 'tanh':
                    col = col * 2 - 1
                data_t.append(col.reshape([-1, 1]))
                self.output_info.append((1, self.act))

            else:
                col_t = np.zeros([len(data), info['size']])
                col_t[np.arange(len(data)), col.astype('int32')] = 1
                data_t.append(col_t)
                self.output_info.append((info['size'], 'softmax'))

        return np.concatenate(data_t, axis=1)

    def inverse_transform(self, data):
        data_t = np.zeros([len(data), len(self.meta)])

        data = data.copy()
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                current = data[:, 0]
                data = data[:, 1:]

                if self.act == 'tanh':
                    current = (current + 1) / 2

                current = np.clip(current, 0, 1)
                data_t[:, id_] = current * (info['max'] - info['min']) + info['min']

            elif info['type'] == ORDINAL:
                current = data[:, 0]
                data = data[:, 1:]

                if self.act == 'tanh':
                    current = (current + 1) / 2

                current = current * info['size']
                current = np.round(current).clip(0, info['size'] - 1)
                data_t[:, id_] = current
            else:
                current = data[:, :info['size']]
                data = data[:, info['size']:]
                data_t[:, id_] = np.argmax(current, axis=1)

        return data_t


class Reconstructor(Module):

    def __init__(self, data_dim, reconstructor_dim, embedding_dim):
        super(Reconstructor, self).__init__()
        dim = data_dim
        seq = []
        for item in list(reconstructor_dim):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq += [Linear(dim, embedding_dim)]
        self.seq = Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


class Discriminator(Module):

    def __init__(self, input_dim, discriminator_dim):
        super(Discriminator, self).__init__()
        dim = input_dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), ReLU(), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def forward(self, input):
        return self.seq(input)


class Generator(Module):

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input, output_info):
        data = self.seq(input)
        data_t = []
        st = 0
        for item in output_info:
            if item[1] == 'tanh':
                ed = st + item[0]
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed

            elif item[1] == 'softmax':
                ed = st + item[0]
                data_t.append(softmax(data[:, st:ed], dim=1))
                st = ed

            else:
                assert 0

        return torch.cat(data_t, dim=1)


class VEEGANSynthesizer(BaseSynthesizer):
    """VEEGANSynthesizer."""

    def __init__(
        self,
        embedding_dim=32,
        gen_dim=(128, 128),
        dis_dim=(128, ),
        rec_dim=(128, 128),
        l2scale=1e-6,
        batch_size=500,
        epochs=300
    ):

        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim
        self.rec_dim = rec_dim

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, train_data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.transformer = GeneralTransformer(act='tanh')
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self.device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        data_dim = self.transformer.output_dim
        self.generator = Generator(self.embedding_dim, self.gen_dim, data_dim).to(self.device)
        discriminator = Discriminator(self.embedding_dim + data_dim, self.dis_dim).to(self.device)
        reconstructor = Reconstructor(data_dim, self.rec_dim, self.embedding_dim).to(self.device)

        optimizer_params = dict(lr=1e-3, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)
        optimizerR = Adam(reconstructor.parameters(), **optimizer_params)

        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1
        for i in range(self.epochs):
            for id_, data in enumerate(loader):
                real = data[0].to(self.device)
                realz = reconstructor(real)
                y_real = discriminator(torch.cat([real, realz], dim=1))

                fakez = torch.normal(mean=mean, std=std)
                fake = self.generator(fakez, self.transformer.output_info)
                fakezrec = reconstructor(fake)
                y_fake = discriminator(torch.cat([fake, fakez], dim=1))

                loss_d = (
                    -(torch.log(torch.sigmoid(y_real) + 1e-4).mean())
                    - (torch.log(1. - torch.sigmoid(y_fake) + 1e-4).mean())
                )

                numerator = -y_fake.mean() + mse_loss(fakezrec, fakez, reduction='mean')
                loss_g = numerator / self.embedding_dim
                loss_r = numerator / self.embedding_dim
                optimizerD.zero_grad()
                loss_d.backward(retain_graph=True)
                optimizerD.step()
                optimizerG.zero_grad()
                loss_g.backward(retain_graph=True)
                optimizerG.step()
                optimizerR.zero_grad()
                loss_r.backward()
                optimizerR.step()

    def sample(self, n):

        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self.device)
            fake = self.generator(noise, output_info)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self.transformer.inverse_transform(data)
