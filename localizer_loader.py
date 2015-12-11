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
Load data for localization
"""
import numpy as np
import math
from neon.data import ImageLoader


class LocalizerLoader(ImageLoader):

    def __init__(self, repo_dir, inner_size, point_num=None,
                 do_transforms=True, rgb=True, multiview=False,
                 set_name='train', subset_pct=100,
                 nlabels=1, macro=True, dtype=np.float32):
        super(LocalizerLoader, self).__init__(
            repo_dir, inner_size, do_transforms, rgb, multiview, set_name,
            subset_pct, nlabels, macro, dtype)
        self.imgheight = inner_size
        self.imgwidth = inner_size
        self.mask = self.be.iobuf(self.imgheight*self.imgwidth, dtype=dtype)
        assert point_num is 1 or point_num is 2
        self.point_num = point_num

    def maketarget(self, idx, xc, yc, width, imgheight, imgwidth):
        xstart = 0 if xc <= width else xc - width
        ystart = 0 if yc <= width else yc - width
        xend = imgwidth if imgwidth - xc <= width else xc + width + 1
        yend = imgheight if imgheight - yc <= width else yc + width + 1
        inds = []
        for x in range(xstart, xend):
            for y in range(ystart, yend):
                dist = math.hypot(xc - x, yc - y)
                if dist >= width:
                    continue
                self.mask[y * imgwidth + x, idx] = (1 - dist / width)

    def maketargets(self, labels):
        self.mask[:] = 0
        for i in range(self.be.bsz):
            xc = labels[0, i]
            yc = labels[1, i]
            self.maketarget(i, xc, yc, self.imgwidth/60,
                            self.imgheight, self.imgwidth)
        return labels

    def __iter__(self):
        for start in range(self.start_idx, self.ndata, self.bsz):
            end = min(start + self.bsz, self.ndata)
            if end == self.ndata:
                self.start_idx = self.bsz - (self.ndata - start)
            self.loaderlib.next(self.loader)
            self.data[:] = self.buffers[self.idx]
            self.data[:] /= 255.
            labels = self.labels[self.idx].get()
            labels = labels[:2] if self.point_num is 1 else labels[2:]
            self.idx = 1 if self.idx == 0 else 0
            if self.set_name == 'train':
                self.maketargets(labels)
                yield self.data, self.mask
            else:
                yield self.data, labels
