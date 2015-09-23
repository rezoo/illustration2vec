from i2v.base import Illustration2VecBase
import json
import numpy as np
from caffe import Classifier
from caffe.io import resize_image


class I2VCaffeExtractor(Illustration2VecBase):

    def _extract(self, inputs, layername):
        shape = (
            len(inputs), self.net.image_dims[0],
            self.net.image_dims[1], inputs[0].shape[2])
        input_ = np.zeros(shape, dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = resize_image(in_, self.net.image_dims)
        # Take center crop.
        center = np.array(self.net.image_dims) / 2.0
        crop = np.tile(center, (1, 2))[0] + np.concatenate([
            -self.net.crop_dims / 2.0,
            self.net.crop_dims / 2.0
        ])
        input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]
        # Classify
        caffe_in = np.zeros(
            np.array(input_.shape)[[0, 3, 1, 2]], dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = \
                self.net.transformer.preprocess(self.net.inputs[0], in_)
        out = self.net.forward_all(
            blobs=[layername], **{self.net.inputs[0]: caffe_in})[layername]
        return out


def make_i2v_with_caffe(net_path, param_path, mean_path, tag_path=None):
    net = Classifier(
        net_path, param_path,
        mean=np.load(mean_path).mean(1).mean(1),
        channel_swap=(2, 1, 0))
    if tag_path is not None:
        tags = json.loads(open(tag_path, "r").read())
        assert(len(tags) == 1539)
        return I2VCaffeExtractor(net, tags)
    else:
        return I2VCaffeExtractor(net)
