from i2v.base import Illustration2VecBase

caffe_available = False
chainer_available = False

try:
    from i2v.caffe_extractor import I2VCaffeExtractor, make_i2v_with_caffe
    caffe_available = True
except ImportError:
    pass

if not any([caffe_available, chainer_available]):
    raise ImportError('i2v requires caffe or chainer package')
