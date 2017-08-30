#!/usr/bin/env sh
# This scripts downloads the pre-trained models.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading pre-trained models..."
wget https://github.com/rezoo/illustration2vec/releases/download/v2.0.0/tag_list.json.gz
wget https://github.com/rezoo/illustration2vec/releases/download/v2.0.0/illust2vec_tag.prototxt
wget https://github.com/rezoo/illustration2vec/releases/download/v2.0.0/illust2vec_tag_ver200.caffemodel
wget https://github.com/rezoo/illustration2vec/releases/download/v2.0.0/illust2vec_ver200.caffemodel
gunzip tag_list.json.gz

echo "Done."
