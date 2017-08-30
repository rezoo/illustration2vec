# Illustration2Vec

``illustration2vec (i2v)`` is a simple library for estimating a set of tags and
extracting semantic feature vectors from given illustrations.
For details, please see
[our main paper](https://github.com/rezoo/illustration2vec/raw/master/papers/illustration2vec-main.pdf).

# Requirements

* Pre-trained models (``i2v`` uses Convolutional Neural Networks. Please download
  several pre-trained models from
  [here](https://github.com/rezoo/illustration2vec/releases),
  or execute ``get_models.sh`` in this repository).
* ``numpy`` and ``scipy``
* ``PIL`` (Python Imaging Library) or its alternatives (e.g., ``Pillow``) 
* ``skimage`` (Image processing library for python)

In addition to the above libraries and the pre-trained models, `i2v` requires
either ``caffe`` or ``chainer`` library. If you are not familiar with deep
learning libraries, we recommend to use ``chainer`` that can be installed
via ``pip`` command.

# How to use

In this section, we show two simple examples -- tag prediction and the the
feature vector extraction -- by using the following illustration [1].

![slide](images/miku.jpg)

[1] Hatsune Miku (初音ミク), © Crypton Future Media, INC.,
http://piapro.net/en_for_creators.html.
This image is licensed under the Creative Commons - Attribution-NonCommercial,
3.0 Unported (CC BY-NC).

## Tag prediction

``i2v`` estimates a number of semantic tags from given illustrations
in the following manner.
```python
import i2v
from PIL import Image

illust2vec = i2v.make_i2v_with_chainer(
    "illust2vec_tag_ver200.caffemodel", "tag_list.json")

# In the case of caffe, please use i2v.make_i2v_with_caffe instead:
# illust2vec = i2v.make_i2v_with_caffe(
#     "illust2vec_tag.prototxt", "illust2vec_tag_ver200.caffemodel",
#     "tag_list.json")

img = Image.open("images/miku.jpg")
illust2vec.estimate_plausible_tags([img], threshold=0.5)
```

``estimate_plausible_tags()`` returns dictionaries that have a pair of
tag and its confidence.
```python
[{'character': [(u'hatsune miku', 0.9999994039535522)],
  'copyright': [(u'vocaloid', 0.9999998807907104)],
  'general': [(u'thighhighs', 0.9956372380256653),
   (u'1girl', 0.9873462319374084),
   (u'twintails', 0.9812833666801453),
   (u'solo', 0.9632901549339294),
   (u'aqua hair', 0.9167950749397278),
   (u'long hair', 0.8817108273506165),
   (u'very long hair', 0.8326570987701416),
   (u'detached sleeves', 0.7448858618736267),
   (u'skirt', 0.6780789494514465),
   (u'necktie', 0.5608364939689636),
   (u'aqua eyes', 0.5527772307395935)],
  'rating': [(u'safe', 0.9785731434822083),
   (u'questionable', 0.020535090938210487),
   (u'explicit', 0.0006299660308286548)]}]
```
These tags are classified into the following four categories:
*general tags* representing general attributes included in an image,
*copyright tags* representing the specific name of the copyright,
*character tags* representing the specific name of the characters,
and *rating tags* representing X ratings.

If you want to focus on several specific tags, use ``estimate_specific_tags()`` instead.
```python
illust2vec.estimate_specific_tags([img], ["1girl", "blue eyes", "safe"])
# -> [{'1girl': 0.9873462319374084, 'blue eyes': 0.01301183458417654, 'safe': 0.9785731434822083}]
```

## Feature vector extraction

``i2v`` can extract a semantic feature vector from an illustration.
```python
import i2v
from PIL import Image

# In the feature vector extraction, you do not need to specify the tag.
illust2vec = i2v.make_i2v_with_chainer("illust2vec_ver200.caffemodel")

# illust2vec = i2v.make_i2v_with_caffe(
#     "illust2vec.prototxt", "illust2vec_ver200.caffemodel")

img = Image.open("images/miku.jpg")

# extract a 4,096-dimensional feature vector
result_real = illust2vec.extract_feature([img])
print("shape: {}, dtype: {}".format(result_real.shape, result_real.dtype))
print(result_real)

# i2v also supports a 4,096-bit binary feature vector
result_binary = illust2vec.extract_binary_feature([img])
print("shape: {}, dtype: {}".format(result_binary.shape, result_binary.dtype))
print(result_binary)
```
The output is the following:
```
shape: (1, 4096), dtype: float32
[[ 7.47459459  3.68610668  0.5379501  ..., -0.14564702  2.71820974
   7.31408596]]
shape: (1, 512), dtype: uint8
[[246 215  87 107 249 190 101  32 187  18 124  90  57 233 245 243 245  54
  229  47 188 147 161 149 149 232  59 217 117 112 243  78  78  39  71  45
  235  53  49  77  49 211  93 136 235  22 150 195 131 172 141 253 220 104
  163 220 110  30  59 182 252 253  70 178 148 152 119 239 167 226 202  58
  179 198  67 117 226  13 204 246 215 163  45 150 158  21 244 214 245 251
  124 155  86 250 183  96 182  90 199  56  31 111 123 123 190  79 247  99
   89 233  61 105  58  13 215 159 198  92 121  39 170 223  79 245  83 143
  175 229 119 127 194 217 207 242  27 251 226  38 204 217 125 175 215 165
  251 197 234  94 221 188 147 247 143 247 124 230 239  34  47 195  36  39
  111 244  43 166 118  15  81 177   7  56 132  50 239 134  78 207 232 188
  194 122 169 215 124 152 187 150  14  45 245  27 198 120 146 108 120 250
  199 178  22  86 175 102   6 237 111 254 214 107 219  37 102 104 255 226
  206 172  75 109 239 189 211  48 105  62 199 238 211 254 255 228 178 189
  116  86 135 224   6 253  98  54 252 168  62  23 163 177 255  58  84 173
  156  84  95 205 140  33 176 150 210 231 221  32  43 201  73 126   4 127
  190 123 115 154 223  79 229 123 241 154  94 250   8 236  76 175 253 247
  240 191 120 174 116 229  37 117 222 214 232 175 255 176 154 207 135 183
  158 136 189  84 155  20  64  76 201  28 109  79 141 188  21 222  71 197
  228 155  94  47 137 250  91 195 201 235 249 255 176 245 112 228 207 229
  111 232 157   6 216 228  55 153 202 249 164  76  65 184 191 188 175  83
  231 174 158  45 128  61 246 191 210 189 120 110 198 126  98 227  94 127
  104 214  77 237  91 235 249  11 246 247  30 152  19 118 142 223   9 245
  196 249 255   0 113   2 115 149 196  59 157 117 252 190 120  93 213  77
  222 215  43 223 222 106 138 251  68 213 163  57  54 252 177 250 172  27
   92 115 104 231  54 240 231  74  60 247  23 242 238 176 136 188  23 165
  118  10 197 183  89 199 220  95 231  61 214  49  19  85  93  41 199  21
  254  28 205 181 118 153 170 155 187  60  90 148 189 218 187 172  95 182
  250 255 147 137 157 225 127 127  42  55 191 114  45 238 228 222  53  94
   42 181  38 254 177 232 150  99]]
```

# License
The pre-trained models and the other files we have provided are licensed
under the MIT License.
