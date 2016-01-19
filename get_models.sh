#!/usr/bin/env sh
# This scripts downloads the pre-trained models.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading pre-trained models..."
wget http://illustration2vec.net/models/tag_list.json.gz
wget http://illustration2vec.net/models/illust2vec_tag.prototxt
wget http://illustration2vec.net/models/illust2vec_tag_ver200.caffemodel
wget http://illustration2vec.net/models/illust2vec.prototxt
wget http://illustration2vec.net/models/illust2vec_ver200.caffemodel
gunzip tag_list.json.gz

echo "Done."
