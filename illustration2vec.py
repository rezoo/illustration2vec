# -*- coding: utf-8 -*-
# ------------------------------------
# python modules
# ------------------------------------
import os
import glob
import os.path
import csv
import pandas as pd
from PIL import Image
# ------------------------------------
# own python modules
# ------------------------------------
import i2v
# ------------------------------------
# Main function
# ------------------------------------

def main(args):
    #引数の読み込み
    print(args)
    src_dir = args.input_dir
    out_dir = args.out_dir
    threshold = float(args.threshold)
    vectorize = args.vectorize
    title = os.path.split(src_dir)[0].split('/')[len(os.path.split(src_dir)[0].split('/')) - 2]
    if vectorize == "False":
        out_file = out_dir + '/' +  title + '_tags' + str(threshold) + '_output.csv'
        f = open(out_file, 'w')
        f.close()
        #read model
        print("start reading tag model...")
        illust2vec = i2v.make_i2v_with_chainer(
            "illust2vec_tag_ver200.caffemodel", "tag_list.json")
        print("finished reading model")
    if vectorize == "True":
        out_file = out_dir + '/' +  title + '_vecs' + '_output.csv'
        f = open(out_file, 'w')
        f.close()
        #read model
        print("start reading vec model...")
        #don't have to use tag model and don't have to include tag_list.json
        illust2vec = i2v.make_i2v_with_chainer(
            "illust2vec_ver200.caffemodel")
        print("finished reading model")
    # In the case of caffe, please use i2v.make_i2v_with_caffe instead:
    # illust2vec = i2v.make_i2v_with_caffe(
    #     "illust2vec_tag.prototxt", "illust2vec_tag_ver200.caffemodel",
    #     "tag_list.json")
    files = []
    jpgs = []
    for x in os.listdir(src_dir):
        if os.path.isfile(src_dir + x):
            files.append(x) 
    for y in files:
        if(y[-4:] == '.jpg'):     #ファイル名の後ろ4文字を取り出してそれが.jpgなら
            jpgs.append(y)  #リストに追加
    for file_name in jpgs:
        file_path = src_dir + '/' + file_name
        vectorize_f(illust2vec, file_path, out_file, threshold, vectorize)

def vectorize_f(illust2vec, file_path, out_file, threshold, vectorize):
    if vectorize == "False":
        img = Image.open(file_path, 'r')
        print(file_path + " opened!")
        result = illust2vec.estimate_plausible_tags([img], threshold = threshold)
        print(file_path + " added tags!")
        result.insert(0, os.path.basename(file_path))
        result.append(list(zip(result[1]["rating"])))
        #print(result)
        f = open(out_file, 'a')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(result)
        f.close()
    if vectorize == "True":
        img = Image.open(file_path, 'r')
        print(file_path + " opened!")
        vec = illust2vec.extract_feature([img])
        print(file_path + " vectorize!!")
        f = open(out_file, 'a')
        df = pd.DataFrame(vec)
        df.rename(index={0: os.path.basename(file_path)}, inplace=True)
        df.to_csv(f, header=False)
        f.close()






        
