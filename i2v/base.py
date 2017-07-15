from abc import ABCMeta, abstractmethod
import numpy as np


class Illustration2VecBase(object):

    __metaclass__ = ABCMeta

    def __init__(self, net, tags=None, threshold=None):
        self.net = net
        if tags is not None:
            self.tags = np.array(tags)
            self.index = {t: i for i, t in enumerate(tags)}
        else:
            self.tags = None

        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = None

    @abstractmethod
    def _extract(self, inputs, layername):
        pass

    def _convert_image(self, image):
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            # convert a monochrome image to a color one
            ret = np.empty((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
            ret[:] = arr.reshape(arr.shape[0], arr.shape[1], 1)
            return ret
        elif arr.ndim == 3:
            # if arr contains alpha channel, remove it
            return arr[:,:,:3]
        else:
            raise TypeError('unsupported image specified')

    def _estimate(self, images):
        assert(self.tags is not None)
        imgs = [self._convert_image(img) for img in images]
        prob = self._extract(imgs, layername='prob')
        prob = prob.reshape(prob.shape[0], -1)
        return prob

    def estimate_specific_tags(self, images, tags):
        prob = self._estimate(images)
        return [{t: float(prob[i, self.index[t]]) for t in tags}
                for i in range(prob.shape[0])]

    def estimate_top_tags(self, images, n_tag=10):
        prob = self._estimate(images)
        general_prob = prob[:, :512]
        character_prob = prob[:, 512:1024]
        copyright_prob = prob[:, 1024:1536]
        rating_prob = prob[:, 1536:]
        general_arg = np.argsort(-general_prob, axis=1)[:, :n_tag]
        character_arg = np.argsort(-character_prob, axis=1)[:, :n_tag]
        copyright_arg = np.argsort(-copyright_prob, axis=1)[:, :n_tag]
        rating_arg = np.argsort(-rating_prob, axis=1)
        result = []
        for i in range(prob.shape[0]):
            result.append({
                'general': list(zip(
                    self.tags[general_arg[i]],
                    general_prob[i, general_arg[i]].tolist())),
                'character': list(zip(
                    self.tags[512 + character_arg[i]],
                    character_prob[i, character_arg[i]].tolist())),
                'copyright': list(zip(
                    self.tags[1024 + copyright_arg[i]],
                    copyright_prob[i, copyright_arg[i]].tolist())),
                'rating': list(zip(
                    self.tags[1536 + rating_arg[i]],
                    rating_prob[i, rating_arg[i]].tolist())),
            })
        return result

    def __extract_plausible_tags(self, preds, f):
        result = []
        for pred in preds:
            general = [(t, p) for t, p in pred['general'] if f(t, p)]
            character = [(t, p) for t, p in pred['character'] if f(t, p)]
            copyright = [(t, p) for t, p in pred['copyright'] if f(t, p)]
            result.append({
                'general': general,
                'character': character,
                'copyright': copyright,
                'rating': pred['rating'],
            })
        return result

    def estimate_plausible_tags(
            self, images, threshold=0.25, threshold_rule='constant'):
        preds = self.estimate_top_tags(images, n_tag=512)
        result = []
        if threshold_rule == 'constant':
            return self.__extract_plausible_tags(
                preds, lambda t, p: p > threshold)
        elif threshold_rule == 'f0.5':
            if self.threshold is None:
                raise TypeError(
                    'please specify threshold option during init.')
            return self.__extract_plausible_tags(
                preds, lambda t, p: p > self.threshold[self.index[t], 0])
        elif threshold_rule == 'f1':
            if self.threshold is None:
                raise TypeError(
                    'please specify threshold option during init.')
            return self.__extract_plausible_tags(
                preds, lambda t, p: p > self.threshold[self.index[t], 1])
        elif threshold_rule == 'f2':
            if self.threshold is None:
                raise TypeError(
                    'please specify threshold option during init.')
            return self.__extract_plausible_tags(
                preds, lambda t, p: p > self.threshold[self.index[t], 2])
        else:
            raise TypeError('unknown rule specified')
        return result

    def extract_feature(self, images):
        imgs = [self._convert_image(img) for img in images]
        feature = self._extract(imgs, layername='encode1')
        feature = feature.reshape(feature.shape[0], -1)
        return feature

    def extract_binary_feature(self, images):
        imgs = [self._convert_image(img) for img in images]
        feature = self._extract(imgs, layername='encode1neuron')
        feature = feature.reshape(feature.shape[0], -1)
        binary_feature = np.zeros_like(feature, dtype=np.uint8)
        binary_feature[feature > 0.5] = 1
        return np.packbits(binary_feature, axis=1)
