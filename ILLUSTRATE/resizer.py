# -*- coding: utf-8 -*- 
 
# ------------------------------------
# python modules
# ------------------------------------
from PIL import Image
import os
# ------------------------------------
# own python modules
# ------------------------------------
 
 
# ------------------------------------
# Main function
# ------------------------------------
def main(args):
    #引数の読み込み
    print(args)
    src_dir = args.input_dir
    out_dir = args.out_dir
    height = int(args.resize)
    resize_all(src_dir, out_dir, height)
#    １ファイルのみを以下でテスト
#    #引数の読み込み
#    file_path = args.inputfile
#    out_dir = args.outdir
#    height = int(args.resize)
#    # 画像をreadonlyで開く
#    img = Image.open(file_path, 'r')
#    # 画像サイズを取得後、リサイズ後の画像ピクセルを計算
#    before_x, before_y = img.size[0], img.size[1]
#    x = int(round(float(height) / float(before_y) * float(before_x)))
#    y = height
#    #imgを一時的に代入
#    resize_img = img
#    #アンチエイリアスありで縮小
#    resize_img.thumbnail((x, y), Image.ANTIALIAS)
#    #アウトプット先を指定
#    outfile = out_dir + os.path.splitext (os.path.basename(file_path))[0] + "_thumbnail.jpg"
#    #サムネイルを保存
#    resize_img.save(outfile, 'jpeg', quality=100)
#    print("RESIZED!:{}[{}x{}] --> {}x{}".format(outfile, before_x, before_y, x, y))
#    #im.save(outfile, "JPEG")
#    #resize(image, )
#    #resize_all(my_src_dir, my_out_dir)
    
def resize_all(src_dir, out_dir, height):
    files = []
    jpgs = []
    for x in os.listdir(src_dir):
        if os.path.isfile(src_dir + x):
            files.append(x)
    for y in files:
        if(y[-4:] == '.jpg'):     #ファイル名の後ろ4文字を取り出してそれが.jpgなら
            jpgs.append(y)  #リストに追加
    for index, file_name in enumerate(jpgs):
        file_path = src_dir + '/' + file_name
        print(file_path)
        number = index
        number_padded = '{0:08d}'.format(number)
        print(number_padded)
        resize(file_path, out_dir, height, number_padded)

def resize(file_path, out_dir, height, number_padded):
    img = Image.open(file_path, 'r')
    # 画像サイズを取得後、リサイズ後の画像ピクセルを計算
    before_x, before_y = img.size[0], img.size[1]
    x = int(round(float(height) / float(before_y) * float(before_x)))
    y = height
    #imgを一時的に代入
    resize_img = img
    #アンチエイリアスありで縮小
    resize_img.thumbnail((x, y), Image.ANTIALIAS)
    #アウトプット先を指定
    out_path = out_dir + str(number_padded) + "_" + os.path.splitext(os.path.basename(file_path))[0] + "_thumbnail.jpg"
    #サムネイルを保存
    resize_img.save(out_path, 'jpeg', quality=100)
    print("RESIZED!:{}[{}x{}] --> {}x{}".format(out_path, before_x, before_y, x, y))
    
def get_dest_size(image, resize_rate):
    (w,h) = image.size
    #w_dest = int(w * resize_rate)
    #h_dest = int(h * resize_rate)
    #return (w, h, w_dest, h_dest)
    return (w, h)
