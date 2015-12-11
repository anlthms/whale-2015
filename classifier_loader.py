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
Load data for classification
"""
from neon.data import ImageLoader


class ClassifierLoader(ImageLoader):

    def __iter__(self):
        for start in range(self.start_idx, self.ndata, self.bsz):
            end = min(start + self.bsz, self.ndata)
            if end == self.ndata:
                self.start_idx = self.bsz - (self.ndata - start)
            self.loaderlib.next(self.loader)
            self.data[:] = self.buffers[self.idx]
            self.data[:] = self.data / 255.
            self.onehot_labels[:] = self.be.onehot(self.labels[self.idx],
                                                   axis=0)
            self.idx = 1 if self.idx == 0 else 0
            yield self.data, self.onehot_labels
