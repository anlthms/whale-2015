import math
import os
import numpy as np
from neon.callbacks.callbacks import Callback
from labels import read_labels


class Evaluator(Callback):

    def __init__(self, callback_data, model, dataset, imwidth,
                 epochs, datadir, pointnum):
        super(Evaluator, self).__init__()
        self.model = model
        self.dataset = dataset
        self.callback_data = callback_data
        self.imwidth = imwidth
        self.epochs = epochs
        self.min_dist = 4 * imwidth
        traindir = os.path.join(os.path.dirname(datadir), 'train')
        _, _, x1map, y1map, x2map, y2map = read_labels(
            traindir, 'points1.json', 'points2.json', imwidth)
        xmap, ymap = (x1map, y1map) if pointnum == 1 else (x2map, y2map)
        self.xymean = np.array([np.mean(xmap.values()),
                                np.mean(ymap.values())])

    def get_xy(self, inds):
        preds = np.empty((inds.shape[0], 2))
        preds[:, 0] = inds % self.imwidth
        preds[:, 1] = inds / self.imwidth
        return preds

    def get_outputs(self):
        self.model.initialize(self.dataset)
        self.dataset.reset()
        preds = None
        for idx, (x, t) in enumerate(self.dataset):
            if preds is None:
                (dim0, dim1) = x.shape
                preds = np.empty((self.dataset.nbatches * dim1, 2),
                                 dtype=x.dtype)
            cur_batch = slice(idx * dim1, (idx + 1) * dim1)
            x = self.model.fprop(x, inference=True)
            probs = x.get().T
            inds = np.argmax(probs, axis=1)
            preds[cur_batch] = self.get_xy(inds)
        return preds[:self.dataset.ndata]

    def on_epoch_end(self, epoch):
        preds = self.get_outputs()
        diffs = preds - self.xymean
        dist = math.hypot(*np.mean(diffs, axis=0))
        dist += np.mean(np.sqrt(np.sum(diffs * diffs, axis=1)))
        print('Heuristic estimate of test error %.2f' % dist)
        if dist < self.min_dist:
            self.min_dist = dist
            if epoch >= self.epochs / 4:
                self.model.finished = True
                print('Stopping early.')
                return
        if epoch == self.epochs - 1:
            # Early stopping did not kick in. We probably didn't converge well.
            print('WARNING: model may not be optimal.')
