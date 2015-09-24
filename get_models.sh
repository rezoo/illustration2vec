#!/usr/bin/env sh
# This scripts downloads the pre-trained models.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading pre-trained models..."
wget --no-check-certificate http://illustration2vec.net/models/tag_list.json.gz
wget --no-check-certificate http://illustration2vec.net/models/illust2vec_tag.prototxt
wget --no-check-certificate http://illustration2vec.net/models/illust2vec_tag_ver200.caffemodel
wget --no-check-certificate http://illustration2vec.net/models/illust2vec.prototxt
wget --no-check-certificate http://illustration2vec.net/models/illust2vec_ver200.caffemodel
gunzip tag_list.json.gz

echo "Done."
