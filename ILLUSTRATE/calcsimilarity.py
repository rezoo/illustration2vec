# -*- coding: utf-8 -*-

# ------------------------------------
# python modules
# ------------------------------------
import pandas as pd
import numpy as np
import scipy.spatial.distance as dis
import math
from PIL import Image, ImageDraw, ImageFont
# ------------------------------------
# own python modules
# ------------------------------------

# ------------------------------------
# Main function
# ------------------------------------

def main(args):
    #引数の読み込み
    print(args)
    inputfile = args.input_file
    out_dir = args.out_dir
    title = args.title
    out_file = out_dir + title + "_cos_similarity.csv"
    df = pd.read_csv(inputfile, header = None)
    df.index = df[0]
    filename = df[0]
    df = df.drop(0, axis=1)
    listAppend(len(df), df, filename, out_file)

def listAppend(n, df, filename, out_file):
    name = pd.DataFrame(filename[0:n])
    name.drop(0, axis=1).T.to_csv(out_file)
    for j in range(n):
        list = []
        for i in range(n):
            list.append('{:.5}'.format(np.round(1 - dis.cosine(df.ix[j, :], df.ix[i, :]), 4)))
        with open(out_file, 'a') as f:
            x = pd.DataFrame(list)
            x = x.T
            x.index = name.ix[j]
            x.colmun = name.ix[0:n]
            x.to_csv(f, header=False)
        print("calculating " + filename[j] + "...")
    return(x)

