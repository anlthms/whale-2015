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
Localize points within images
"""

import sys
import os
import json
import numpy as np
from neon.util.argparser import NeonArgparser
from neon.initializers import Gaussian
from neon.layers import Conv, Deconv, GeneralizedCost
from neon.optimizers import Adadelta
from neon.transforms import Rectlin, Logistic, SumSquared
from neon.models import Model
from neon.callbacks.callbacks import Callbacks
from localizer_loader import LocalizerLoader
from evaluator import Evaluator


parser = NeonArgparser(__doc__)
parser.add_argument('-tw', '--test_data_dir',
                    default='',
                    help='directory in which to find test images')
parser.add_argument('-pn', '--point_num', default=None, help='1 or 2')
parser.add_argument('-iw', '--image_width', default=384, help='image width')
args = parser.parse_args()
point_num = int(args.point_num)
imwidth = int(args.image_width)

train = LocalizerLoader(repo_dir=args.data_dir, inner_size=imwidth,
                        set_name='train', nlabels=4, do_transforms=False,
                        point_num=point_num)
test = LocalizerLoader(repo_dir=args.test_data_dir, inner_size=imwidth,
                       set_name='validation', nlabels=4, do_transforms=False,
                       point_num=point_num)
train.init_batch_provider()
test.init_batch_provider()
init = Gaussian(scale=0.1)
opt = Adadelta(decay=0.9)
common = dict(init=init, batch_norm=True, activation=Rectlin())

# Set up the model layers
layers = []
nchan = 128
layers.append(Conv((2, 2, nchan), strides=2, **common))
for idx in range(16):
    layers.append(Conv((3, 3, nchan), **common))
    if nchan > 16:
        nchan /= 2
for idx in range(15):
    layers.append(Deconv((3, 3, nchan), **common))
layers.append(Deconv((4, 4, nchan), strides=2, **common))
layers.append(Deconv((3, 3, 1), init=init, activation=Logistic(shortcut=True)))

cost = GeneralizedCost(costfunc=SumSquared())
mlp = Model(layers=layers)
callbacks = Callbacks(mlp, train, **args.callback_args)
evaluator = Evaluator(callbacks.callback_data, mlp, test, imwidth, args.epochs,
                      args.data_dir, point_num)
callbacks.add_callback(evaluator)
mlp.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost,
        callbacks=callbacks)
train.exit_batch_provider()

preds = evaluator.get_outputs()
paths = np.genfromtxt(os.path.join(args.test_data_dir, 'val_file.csv'),
                      dtype=str)[1:]
basenames = [os.path.basename(path) for path in paths]
filenames = [path.split(',')[0] for path in basenames]
filenames.sort()
content = []
for i, filename in enumerate(filenames):
    item = {
        "annotations":
        [
            {
                "class": "point",
                "x": int(preds[i, 0]),
                "y": int(preds[i, 1])
            }
        ],
        "class": "image",
        "filename": filename
        }
    content.append(item)

json.dump(content, file('testpoints' + args.point_num + '.json', 'w'),
          indent=4)
test.exit_batch_provider()
