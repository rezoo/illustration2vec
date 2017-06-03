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
    image_save_path = out_dir + title + "_similar_"
    image_path = "./inputs/" + title + "/thumbnail/"
    df = pd.read_csv(inputfile, index_col = 0)
    df = df.ix[0:len(df), 0:len(df)]
    df.columns = df.index
    #少ない数でまずやる。999番目の類似度を高いものから１０個出す
    #sort_df = df.ix[:, 999].sort_values(0, ascending=False).head(10)

    #rangeの前後のリストを作成（100個単位を作成)
    num = math.floor(len(df) / 100)
    add_bin = np.array([100, 100])
    first = np.array([0, 100])
    bin = []
    bin.append(first)
    bin_list = []
    next = first
    for k in range(num - 1):
        next = next + add_bin
        bin.append(next)
    bin_df = pd.DataFrame(bin)
    bin_be = list(bin_df[0])
    bin_af = list(bin_df[1])
    bin_be.append((num)  * 100)
    bin_af.append(len(df))

    #画像を100枚単位に10位まで似ている画像を描画
    for (be, af) in zip(bin_be, bin_af):
        #ファイルのpathと類似度をそれぞれリストへ
        filedir = []
        similar_num = []
        number = []
        for i in range(be, af):
            sort_df = df.ix[:, i].sort_values(0, ascending=False).head(10)
            for j in range(len(sort_df)):
                filedir.append(image_path + sort_df.index[j])
                similar_num.append('{:.5}'.format(sort_df.ix[j, 1]))
                number.append(sort_df.index[j][3:])    
        i=0
        j=1
        k=0
        h = 156 #水平ピクセル
        v = 200 #垂直ピクセル
        fontsize = 10
        # マージに利用する下地画像を作成する
        filesize = len(filedir)
        canvas = Image.new('RGB', (10*h, 10*v*math.ceil((filesize/100))), (255, 255, 255))
        for (code, sn, no) in zip(filedir, similar_num, number) :
            i+=1
            k=i-filesize*(j-1)
            print(code, k)
            paste_info(canvas, code, sn, no, fontsize, h, v, k)
            if i==filesize*j:
                # 保存
                canvas.save(image_save_path + str(math.ceil(af/100)) + ".jpg", 'JPEG', quality=100, optimize=True)
                canvas.close()

def paste_info(canvas, code, similar, number, fontsize, h, v, k):
    opacity = 10000  # 透かし文字の透明度を定義します。
    color = (0, 0, 256)  # 透かし文字の色を定義します。
    img = Image.open(code).convert('RGBA') 
    imgr = img.resize((h, v))
    # テキストを描画する画像オブジェクトを作成します。
    txt = Image.new('RGBA', imgr.size, (256, 256, 256, 0))
    draw = ImageDraw.Draw(txt)
    # フォントを取得します。
    fnt = ImageFont.truetype(font='/System/Library/Fonts/Hiragino Sans GB W3.ttc', size=fontsize)
    # number(ファイルの番号を取得)
    
    # 透かし文字の横幅、縦幅を取得します。
    textw, texth = draw.textsize(similar, font=fnt)
    # 透かし文字を中央に入れます。
    draw.text(((imgr.width - textw) / 10, (imgr.height - texth) / 1.1),
          number, font=fnt, fill=color + (opacity,))
    draw.text(((imgr.width - textw) / 10, (imgr.height - texth) / 1),
          similar, font=fnt, fill=color + (opacity,))
    # 画像オブジェクトを重ねます。
    out = Image.alpha_composite(imgr, txt)
    canvas.paste(out, (((k-1)%10)*h, ((k-1)//10)*v))
