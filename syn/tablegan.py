import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import KBinsDiscretizer
import torch
from torch.nn import (BatchNorm2d, Conv2d, ConvTranspose2d, LeakyReLU, Module, ReLU, Sequential, Sigmoid, Tanh, init)
from torch.nn.functional import binary_cross_entropy_with_logits
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

class TableganTransformer(Transformer):

    def __init__(self, side):
        self.height = side

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.meta = self.get_metadata(data, categorical_columns, ordinal_columns)
        self.minn = np.zeros(len(self.meta))
        self.maxx = np.zeros(len(self.meta))
        for i in range(len(self.meta)):
            if self.meta[i]['type'] == CONTINUOUS:
                self.minn[i] = self.meta[i]['min'] - 1e-3
                self.maxx[i] = self.meta[i]['max'] + 1e-3
            else:
                self.minn[i] = -1e-3
                self.maxx[i] = self.meta[i]['size'] - 1 + 1e-3

    def transform(self, data):
        data = data.copy().astype('float32')
        data = (data - self.minn) / (self.maxx - self.minn) * 2 - 1
        if self.height * self.height > len(data[0]):
            padding = np.zeros((len(data), self.height * self.height - len(data[0])))
            data = np.concatenate([data, padding], axis=1)

        return data.reshape(-1, 1, self.height, self.height)

    def inverse_transform(self, data):
        data = data.reshape(-1, self.height * self.height)

        data_t = np.zeros([len(data), len(self.meta)])

        for id_, info in enumerate(self.meta):
            numerator = (data[:, id_].reshape([-1]) + 1)
            data_t[:, id_] = (numerator / 2) * (self.maxx[id_] - self.minn[id_]) + self.minn[id_]
            if info['type'] in [CATEGORICAL, ORDINAL]:
                data_t[:, id_] = np.round(data_t[:, id_])

        return data_t

class Discriminator(Module):
    def __init__(self, meta, side, layers):
        super(Discriminator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        # self.layers = layers

    def forward(self, input):
        return self.seq(input)


class Generator(Module):
    def __init__(self, meta, side, layers):
        super(Generator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        # self.layers = layers

    def forward(self, input_):
        return self.seq(input_)


class Classifier(Module):
    def __init__(self, meta, side, layers, device):
        super(Classifier, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        self.valid = True
        if meta[-1]['name'] != 'label' or meta[-1]['type'] != CATEGORICAL or meta[-1]['size'] != 2:
            self.valid = False

        masking = np.ones((1, 1, side, side), dtype='float32')
        index = len(self.meta) - 1
        self.r = index // side
        self.c = index % side
        masking[0, 0, self.r, self.c] = 0
        self.masking = torch.from_numpy(masking).to(device)

    def forward(self, input):
        label = (input[:, :, self.r, self.c].view(-1) + 1) / 2
        input = input * self.masking.expand(input.size())
        return self.seq(input).view(-1), label


def determine_layers(side, random_dim, num_channels):
    assert side >= 4 and side <= 32

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]
    layers_D += [
        Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0),
        Sigmoid()
    ]

    layers_G = [
        ConvTranspose2d(
            random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)
    ]

    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(prev[0]),
            ReLU(True),
            ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)
        ]
    layers_G += [Tanh()]

    layers_C = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_C += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]

    layers_C += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0)]

    return layers_D, layers_G, layers_C


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)


class TableganSynthesizer(BaseSynthesizer):
    """docstring for TableganSynthesizer??"""

    def __init__(self,
                 random_dim=100,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=500,
                 epochs=300):

        self.random_dim = random_dim
        self.num_channels = num_channels
        self.l2scale = l2scale

        self.batch_size = batch_size
        self.epochs = epochs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        sides = [4, 8, 16, 24, 32]
        for i in sides:
            if i * i >= data.shape[1]:
                self.side = i
                break

        self.transformer = TableganTransformer(self.side)
        self.transformer.fit(data, categorical_columns, ordinal_columns)
        data = self.transformer.transform(data)
        data = torch.from_numpy(data.astype('float32')).to(self.device)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        layers_D, layers_G, layers_C = determine_layers(
            self.side, self.random_dim, self.num_channels)

        self.generator = Generator(self.transformer.meta, self.side, layers_G).to(self.device)
        discriminator = Discriminator(self.transformer.meta, self.side, layers_D).to(self.device)
        classifier = Classifier(
            self.transformer.meta, self.side, layers_C, self.device).to(self.device)

        optimizer_params = dict(lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)
        optimizerC = Adam(classifier.parameters(), **optimizer_params)

        self.generator.apply(weights_init)
        discriminator.apply(weights_init)
        classifier.apply(weights_init)

        for i in range(self.epochs):
            for id_, data in enumerate(loader):
                real = data[0].to(self.device)
                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)

                optimizerD.zero_grad()
                y_real = discriminator(real)
                y_fake = discriminator(fake)
                loss_d = (
                    -(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))
                loss_d.backward()
                optimizerD.step()

                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
                optimizerG.zero_grad()
                y_fake = discriminator(fake)
                loss_g = -(torch.log(y_fake + 1e-4).mean())
                loss_g.backward(retain_graph=True)
                loss_mean = torch.norm(torch.mean(fake, dim=0) - torch.mean(real, dim=0), 1)
                loss_std = torch.norm(torch.std(fake, dim=0) - torch.std(real, dim=0), 1)
                loss_info = loss_mean + loss_std
                loss_info.backward()
                optimizerG.step()

                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
                if classifier.valid:
                    real_pre, real_label = classifier(real)
                    fake_pre, fake_label = classifier(fake)

                    loss_cc = binary_cross_entropy_with_logits(real_pre, real_label)
                    loss_cg = binary_cross_entropy_with_logits(fake_pre, fake_label)

                    optimizerG.zero_grad()
                    loss_cg.backward()
                    optimizerG.step()

                    optimizerC.zero_grad()
                    loss_cc.backward()
                    optimizerC.step()
                    loss_c = (loss_cc, loss_cg)
                else:
                    loss_c = None

                if((id_ + 1) % 50 == 0):
                    print("epoch", i + 1, "step", id_ + 1, loss_d, loss_g, loss_c)

    def sample(self, n):

        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
            fake = self.generator(noise)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        return self.transformer.inverse_transform(data[:n])