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
Write macro batches of data
"""

import numpy as np
import functools
import os
import tarfile
import struct
from glob import glob
from multiprocessing import Pool
from PIL import Image as Image
from neon.util.compat import range, StringIO
from neon.util.persist import save_obj
from neon.util.argparser import NeonArgparser
from labels import read_labels


parser = NeonArgparser(__doc__)
parser.add_argument('--image_dir', help='Directory to find images',
                    required=True)
parser.add_argument('--target_size', type=int, default=384,
                    help='Size in pixels to scale images')
parser.add_argument('--points1_file',
                    help='json file with co-ordinates of point1', default=None)
parser.add_argument('--points2_file',
                    help='json file with co-ordinates of point2', default=None)
parser.add_argument('--id_label', type=int, default=0,
                    help='Whether the labels are IDs')
parser.add_argument('--val_pct', type=int, default=20,
                    help='Validation set percentage')
args = parser.parse_args()


def proc_img(target_size, imgfile=None):
    im = Image.open(imgfile)
    scale_factor = target_size / np.float32(min(im.size))
    filt = Image.BICUBIC if scale_factor > 1 else Image.ANTIALIAS
    im = im.resize((target_size, target_size), filt)
    buf = StringIO()
    im.save(buf, format='JPEG', quality=95)
    return buf.getvalue()


class BatchWriter(object):

    def __init__(self, out_dir, image_dir, points1_file, points2_file,
                 id_label, target_size, val_pct=20,
                 class_samples_max=None, file_pattern='*.jpg',
                 macro_size=3072):
        self.out_dir = os.path.expanduser(out_dir)
        self.image_dir = os.path.expanduser(image_dir)
        self.macro_size = macro_size
        self.num_workers = 8
        self.target_size = target_size
        self.file_pattern = file_pattern
        self.class_samples_max = class_samples_max
        self.val_frac = val_pct / 100.
        self.train_file = os.path.join(self.out_dir, 'train_file.csv')
        self.val_file = os.path.join(self.out_dir, 'val_file.csv')
        self.meta_file = os.path.join(self.out_dir, 'dataset_cache.pkl')
        self.batch_prefix = 'data_batch_'
        self.points1_file = points1_file
        self.points2_file = points2_file
        self.id_label = id_label

    def write_csv_files(self):
        files = glob(os.path.join(self.image_dir, '*.jpg'))
        files.sort()
        if self.val_frac != 1.0:
            filemap, idmap, x1map, y1map, x2map, y2map = (
                read_labels(self.image_dir,
                            self.points1_file,
                            self.points2_file,
                            self.target_size))
        if self.id_label == 1:
            self.label_names = ['id']
        else:
            self.label_names = ['x1', 'y1', 'x2', 'y2']

        indexes = range(len(self.label_names))
        self.label_dict = {k: v for k, v in zip(self.label_names, indexes)}

        tlines = []
        vlines = []

        np.random.shuffle(files)
        v_idx = int(self.val_frac * len(files))
        tfiles = files[v_idx:]
        vfiles = files[:v_idx]
        vfiles.sort()
        if self.id_label == 1:
            if self.val_frac == 1.0:
                vlines = [(f, 0) for f in vfiles]
            else:
                tlines = [(f, idmap[filemap[f]]) for f in tfiles]
        else:
            if self.val_frac == 1.0:
                vlines = [(f, 0, 0, 0, 0) for f in vfiles]
            else:
                tlines = [(f, x1map[f], y1map[f],
                           x2map[f], y2map[f]) for f in tfiles]
        np.random.shuffle(tlines)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        for ff, ll in zip([self.train_file, self.val_file], [tlines, vlines]):
            with open(ff, 'wb') as f:
                if self.id_label == 1:
                    f.write('filename,id\n')
                    for tup in ll:
                        f.write('{},{}\n'.format(*tup))
                else:
                    f.write('filename,x,y\n')
                    for tup in ll:
                        f.write('{},{},{},{},{}\n'.format(*tup))

        self.train_nrec = len(tlines)
        self.ntrain = -(-self.train_nrec // self.macro_size)
        self.train_start = 0

        self.val_nrec = len(vlines)
        self.nval = -(-self.val_nrec // self.macro_size)
        if self.ntrain == 0:
            self.val_start = 100
        else:
            self.val_start = 10 ** int(np.log10(self.ntrain * 10))

    def parse_file_list(self, infile):
        if self.id_label == 1:
            lines = np.loadtxt(infile, delimiter=',',
                               skiprows=1,
                               dtype={'names': ('fname', 'id'),
                                      'formats': (object, 'i4')})
            imfiles = [l[0] for l in lines]
            labels = {'id': [l[1] for l in lines]}
            self.nclass = 447
        else:
            lines = np.loadtxt(
                infile, delimiter=',', skiprows=1,
                dtype={'names': ('fname', 'x1', 'y1', 'x2', 'y2'),
                       'formats': (object, 'i4', 'i4', 'i4', 'i4')})
            imfiles = [l[0] for l in lines]
            labels = {'x1': [l[1] for l in lines], 'y1': [l[2] for l in lines],
                      'x2': [l[3] for l in lines], 'y2': [l[4] for l in lines]}
            self.nclass = 4
        return imfiles, labels

    def write_batches(self, name, offset, labels, imfiles):
        pool = Pool(processes=self.num_workers)
        npts = -(-len(imfiles) // self.macro_size)
        starts = [i * self.macro_size for i in range(npts)]
        imfiles = [imfiles[s:s + self.macro_size] for s in starts]
        labels = [{k: v[s:s + self.macro_size] for k,
                  v in labels.iteritems()} for s in starts]

        print("Writing %d %s batches..." % (len(imfiles), name))
        for i, jpeg_file_batch in enumerate(imfiles):
            proc_img_func = functools.partial(proc_img, self.target_size)
            jpeg_strings = pool.map(proc_img_func, jpeg_file_batch)
            bfile = os.path.join(
                self.out_dir, '%s%d' % (self.batch_prefix, offset + i))
            self.write_binary(jpeg_strings, labels[i], bfile)
        pool.close()

    def write_binary(self, jpegs, labels, ofname):
        num_imgs = len(jpegs)
        if self.id_label == 1:
            keylist = ['id']
        else:
            keylist = ['x1', 'y1', 'x2', 'y2']
        with open(ofname, 'wb') as f:
            f.write(struct.pack('I', num_imgs))
            f.write(struct.pack('I', len(keylist)))

            for key in keylist:
                ksz = len(key)
                f.write(struct.pack('L' + 'B' * ksz, ksz, *bytearray(key)))
                f.write(struct.pack('I' * num_imgs, *labels[key]))

            for i in range(num_imgs):
                jsz = len(jpegs[i])
                bin = struct.pack('I' + 'B' * jsz, jsz, *bytearray(jpegs[i]))
                f.write(bin)

    def save_meta(self):
        save_obj({'ntrain': self.ntrain,
                  'nval': self.nval,
                  'train_start': self.train_start,
                  'val_start': self.val_start,
                  'macro_size': self.macro_size,
                  'batch_prefix': self.batch_prefix,
                  'global_mean': self.global_mean,
                  'label_dict': self.label_dict,
                  'label_names': self.label_names,
                  'val_nrec': self.val_nrec,
                  'train_nrec': self.train_nrec,
                  'img_size': self.target_size,
                  'nclass': self.nclass}, self.meta_file)

    def run(self):
        self.write_csv_files()
        if self.val_frac == 0.0:
            namelist = ['train']
            filelist = [self.train_file]
            startlist = [self.train_start]
        elif self.val_frac == 1.0:
            namelist = ['validation']
            filelist = [self.val_file]
            startlist = [self.val_start]
        else:
            namelist = ['train', 'validation']
            filelist = [self.train_file, self.val_file]
            startlist = [self.train_start, self.val_start]
        for sname, fname, start in zip(namelist, filelist, startlist):
            if fname is not None and os.path.exists(fname):
                imgs, labels = self.parse_file_list(fname)
                if len(imgs) > 0:
                    self.write_batches(sname, start, labels, imgs)
            else:
                print("Skipping %s, file missing" % (sname))
        self.global_mean = np.empty((3, 1))
        self.global_mean[:] = 127
        self.save_meta()


if __name__ == "__main__":
    np.random.seed(0)
    bw = BatchWriter(out_dir=args.data_dir, image_dir=args.image_dir,
                     target_size=args.target_size, macro_size=256,
                     file_pattern='*.jpg',
                     points1_file=args.points1_file,
                     points2_file=args.points2_file,
                     id_label=args.id_label,
                     val_pct=args.val_pct)
    bw.run()
