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
Read labels
"""
import numpy as np
import json
import os
from PIL import Image


def read_labels(traindir, points1_file, points2_file, imwidth):
    datadir = os.path.dirname(traindir)
    train = np.genfromtxt(os.path.join(datadir, 'train.csv'), delimiter=',',
                          skip_header=1, dtype=str)
    nrows = train.shape[0]
    paths = [os.path.join(traindir, train[i, 0]) for i in range(nrows)]
    filemap = {}
    # Make a mapping from filename to id.
    for i in range(len(paths)):
        filemap[paths[i]] = train[i, 1]
    idents = np.unique(train[:, 1])
    # Make a mapping from id to numeric label.
    idmap = {}
    i = 0
    for ident in idents:
        idmap[ident] = i
        i += 1
    if points1_file is None or points2_file is None:
        return filemap, idmap, None, None, None, None

    # Read annotations
    xmap = [{}, {}]
    ymap = [{}, {}]
    for idx in range(2):
        points_file = [points1_file, points2_file][idx]
        assert os.path.exists(points_file)
        points = json.load(file(points_file))
        for point in points:
            assert len(point['annotations']) == 1
            path = os.path.join(traindir, point['filename'])
            im = Image.open(path)
            width, height = im.size
            xmap[idx][path] = int(
                1.0 * point['annotations'][0]['x'] * imwidth / width)
            ymap[idx][path] = int(
                1.0 * point['annotations'][0]['y'] * imwidth / height)
    return filemap, idmap, xmap[0], ymap[0], xmap[1], ymap[1]
