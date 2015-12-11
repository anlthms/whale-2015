#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Classify images
"""

import os
import gzip
import numpy as np
from neon.util.argparser import NeonArgparser
from neon.initializers import Gaussian
from neon.layers import Conv, DropoutBinary, Pooling, GeneralizedCost, Affine
from neon.optimizers import Adadelta
from neon.transforms import Rectlin, Softmax, CrossEntropyMulti
from neon.models import Model
from neon.callbacks.callbacks import Callbacks
from classifier_loader import ClassifierLoader


parser = NeonArgparser(__doc__)
parser.add_argument('-tw', '--test_data_dir', default='',
                    help='directory in which to find test images')
parser.add_argument('-iw', '--image_width', default=384, help='image width')
args = parser.parse_args()
imwidth = int(args.image_width)

train = ClassifierLoader(repo_dir=args.data_dir, inner_size=imwidth,
                         set_name='train', do_transforms=False)
train.init_batch_provider()
init = Gaussian(scale=0.01)
opt = Adadelta(decay=0.9)
common = dict(init=init, batch_norm=True, activation=Rectlin())

layers = []
nchan = 64
layers.append(Conv((2, 2, nchan), strides=2, **common))
for idx in range(6):
    if nchan > 1024:
        nchan = 1024
    layers.append(Conv((3, 3, nchan), strides=1, **common))
    layers.append(Pooling(2, strides=2))
    nchan *= 2
layers.append(DropoutBinary(keep=0.2))
layers.append(Affine(nout=447, init=init, activation=Softmax()))

cost = GeneralizedCost(costfunc=CrossEntropyMulti())
mlp = Model(layers=layers)
callbacks = Callbacks(mlp, train, **args.callback_args)
mlp.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost,
        callbacks=callbacks)
train.exit_batch_provider()

test = ClassifierLoader(repo_dir=args.test_data_dir, inner_size=imwidth,
                        set_name='validation', do_transforms=False)
test.init_batch_provider()
probs = mlp.get_outputs(test)
test.exit_batch_provider()

filcsv = np.loadtxt(os.path.join(args.test_data_dir, 'val_file.csv'),
                    delimiter=',', skiprows=1, dtype=str)
files = [os.path.basename(row[0]) for row in filcsv]
datadir = os.path.dirname(args.data_dir)

with open(os.path.join(datadir, 'sample_submission.csv'), 'r') as fd:
    header = fd.readline()

with gzip.open('subm.csv.gz', 'wb') as fd:
    fd.write(header)
    for i in range(probs.shape[0]):
        fd.write('{},'.format(files[i]))
        row = probs[i].tolist()
        fd.write(','.join(['{:.3e}'.format(elem) for elem in row]))
        fd.write('\n')
