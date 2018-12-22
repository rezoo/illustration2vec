# File to test out the illustration2vec file :

import caffe
import os
from i2v.caffe_i2v import make_i2v_with_caffe
from PIL import Image
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--img_path',
					type=str,
					required=False,
					help='Enter the path to the image',
					default='images/miku.jpg'
					)

parser.add_argument('--model_dir',
					type=str,
					required=False,
					help='Enter the path to the models folder',
					default='models/'
					)
args = parser.parse_args()

# If you have caffe with the gpu installed, use this. Otherwise comment
# the line below.
caffe.set_mode_gpu()

# Download the model file, prototext and the tag list using the script
# here - https://github.com/rezoo/illustration2vec/get_models.sh and
# store it in a directory called models.

model_def = os.path.join(args.model_dir,'illust2vec_tag.prototxt')
model_wieghts = os.path.join(args.model_dir,'illust2vec_tag_ver200.caffemodel')
tag_path = os.path.join(args.model_dir,'tag_list.json')

net = make_i2v_with_caffe(model_def,model_wieghts,tag_path=tag_path)

img = Image.open(args.img_path)

print net.estimate_top_tags([img])
# Take a look at i2v.base.py for other methods for the "caffe_i2v" class.
