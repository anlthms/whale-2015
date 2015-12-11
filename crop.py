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
Crop images
"""
import numpy as np
import json
import sys
import os
import math
import functools
import multiprocessing
from skimage import io
from skimage import transform as tf
from multiprocessing import Pool


def load(bonnetfile, blowholefile):
    bonnets = json.load(file(bonnetfile))
    blowholes = json.load(file(blowholefile))
    return bonnets, blowholes


def crop(path, bonnet, blowhole):
    im = io.imread(path).astype(np.uint8)
    if doscale == 1:
        bonnet['y'] *= float(im.shape[0]) / imwidth
        bonnet['x'] *= float(im.shape[1]) / imwidth
        blowhole['y'] *= float(im.shape[0]) / imwidth
        blowhole['x'] *= float(im.shape[1]) / imwidth
    y = bonnet['y'] - blowhole['y']
    x = bonnet['x'] - blowhole['x']
    dist = math.hypot(x, y)
    minh = 10
    minw = 20
    croph = int((im.shape[0] - 1.0 * dist) // 2)
    cropw = int((im.shape[1] - 2.0 * dist) // 2)
    newh = im.shape[0] - 2 * croph
    neww = im.shape[1] - 2 * cropw
    if croph <= 0 or cropw <= 0 or newh < minh or neww < minw:
        print(' %s unchanged' % os.path.basename(path))
    else:
        angle = math.atan2(y, x) * 180 / math.pi
        centery = 0.4 * bonnet['y'] + 0.6 * blowhole['y']
        centerx = 0.4 * bonnet['x'] + 0.6 * blowhole['x']
        center = (centerx, centery)
        im = tf.rotate(im, angle, resize=False, center=center,
                       preserve_range=True)
        imcenter = (im.shape[1] / 2, im.shape[0] / 2)
        trans = (center[0] - imcenter[0], center[1] - imcenter[1])
        tform = tf.SimilarityTransform(translation=trans)
        im = tf.warp(im, tform)
        im = im[croph:-croph, cropw:-cropw]
    path = os.path.join(dstdir, os.path.basename(path))
    io.imsave(path, im.astype(np.uint8))
    return im.shape[0], im.shape[1]


def cropbatch(points1, points2, count, maxind, index):
    start = index * count
    end = min(start+count, maxind)
    for i in range(start, end):
        point1 = points1[i]
        point2 = points2[i]
        path = os.path.join(srcdir, point1['filename'])
        height, width = crop(path, point1['annotations'][0],
                             point2['annotations'][0])


if len(sys.argv) < 7:
    print('Usage: %s bonnet-points blowhole-points srcdir dstdir '
          'imwidth scale (0/1)' % sys.argv[0])
    sys.exit(0)
srcdir = sys.argv[3]
dstdir = sys.argv[4]
imwidth = int(sys.argv[5])
doscale = int(sys.argv[6])
bonnets, blowholes = load(sys.argv[1], sys.argv[2])
assert len(bonnets) == len(blowholes)
if os.path.exists(dstdir) is False:
    os.mkdir(dstdir)
maxind = len(bonnets)
pcount = multiprocessing.cpu_count()
count = (maxind - 1) / pcount + 1
cropfunc = functools.partial(cropbatch, bonnets, blowholes, count, maxind)
pool = Pool(processes=pcount)
pool.map(cropfunc, range(pcount))
