# -*- coding: utf-8 -*- 
 
# ------------------------------------
# python modules
# ------------------------------------
 
import os
import sys
import argparse as ap
import tempfile
 
# ------------------------------------
# own python modules
# ------------------------------------
# 
# module_dir/module_name.py -> form module_dir.module_name import method
from ILLUSTRATE.constants import *
# ------------------------------------
# Main function
# ------------------------------------
 
def main():
    # Parse options...
    argparser = prepare_argparser()
    args = argparser.parse_args()
#    if args.out_dir:
#       # use a output directory to store UMIduplicates output
#        if not os.path.exists( args.out_dir ):
#            try:
#                os.makedirs( args.out_dir )
#            except:
#                sys.exit( "Output directory (%s) could not be created. Terminating program." % args.out_dir )
    subcommand  = args.subcommand_name
    #resize
    if subcommand == "resize":
        from ILLUSTRATE.resizer import main
        main(args)
    if subcommand == "vectorize":
        from illustration2vec import main
        main(args)
    if subcommand == "calculate":
        from ILLUSTRATE.calcsimilarity import main
        main(args)
    if subcommand == "search":
        from ILLUSTRATE.searchllustration import main
        main(args)
 
def prepare_argparser ():
    """Prepare optparser object. New options will be added in this
    function first.
    """
    #%(prog)sはプログラム名の取得
    #sys.argv[0]でもargparse.ArgumentParser(prog='hoge.py')などでも同じことができる
    description = "%(prog)s -- Search similar illustration using illustration2vec and cosine similarity"
    epilog = "For command line options of each command, type: %(prog)s COMMAND -h"
    #Check community site:
    #Source code: 
    # top-level parser
    argparser = ap.ArgumentParser(description = description, epilog = epilog) #, usage = usage )
    argparser.add_argument("--version", action="version", version="%(prog)s " + VERSION)
    subparsers = argparser.add_subparsers(dest = 'subcommand_name' ) #help="sub-command help")
 
    # command for 'resize'
    add_resize_parser(subparsers)
    # command for 'vectorize'
    add_vectorize_parser(subparsers)
    # command for 'calculate'
    add_calculate_parser(subparsers)
    # command for 'search'
    add_search_parser(subparsers)
    return argparser
 
def add_outdir_option (parser):
    parser.add_argument("--outdir", dest = "out_dir", type = str, default = '',
                        help = "If specified all output files will be written to that directory. Default: the current working directory")
def add_output_group (parser, required = True ):
    output_group = parser.add_mutually_exclusive_group( required = required )
    output_group.add_argument("-o", "--ofile", dest = "ofile", type = str,
                               help = "Output file name. Mutually exclusive with --o-prefix.")
    output_group.add_argument("--o-prefix", dest = "oprefix", type = str,
                               help = "Output file prefix. Mutually exclusive with -o/--ofile.")
 
def add_resize_parser(subparsers):
    """Add main function 'resize' argument parsers.
    """
    argparser_resize = subparsers.add_parser("resize", help = "Resize given image files.")
    # group for input files
    group_input = argparser_resize.add_argument_group("Input files arguments")
    group_input.add_argument("-i", "--input", dest = "input_dir", type = str, required = True,
                              help = "Set input dir must be directory included only jpeg or png files. REQUIRED.")
    group_input.add_argument("-s", "--size", dest = "resize", type = str, required = True,
                              help = "Set height size you want. Aspect ratio is fixed. REQUIRED.")
    # group for output files
    group_output = argparser_resize.add_argument_group("Output arguments")
    add_outdir_option(group_output)
    group_output.add_argument("-n", "--name", dest = "name", type = str,
                               help = "Project name, which will be used to generate output file names. DEFAULT: \"NA\"",
                               default = "NA")
def add_vectorize_parser(subparsers):
    """Add main function 'vectorize' argument parsers.
    """   
    argparser_vectorize = subparsers.add_parser("vectorize", help = "Vectorize given image files.")
    # group for input files
    group_input = argparser_vectorize.add_argument_group("Input files arguments")
    group_input.add_argument("-i", "--input", dest = "input_dir", type = str, required = True,
                              help = "Input file must be resize image files. REQUIRED.")
    group_input.add_argument("-t", "--threshold", dest = "threshold", type = str, required = True,
                              help = "Set thresh hold you want. REQUIRED.")
    group_input.add_argument("-v", "--vectorize", dest = "vectorize", type = str, required = True,
            help = "Set tag (False) or vector (True) you want. REQUIRED.")

    # group for output files
    group_output = argparser_vectorize.add_argument_group("Output arguments")
    add_outdir_option(group_output)
    group_output.add_argument("-n", "--name", dest = "name", type = str,
                               help = "Project name, which will be used to generate output file names. DEFAULT: \"NA\"",
                               default = "NA")
def add_calculate_parser(subparsers):
    """Add main function 'calculate' argument parsers.
    """   
    argparser_caluculate = subparsers.add_parser("calculate", help = "Calculate cosine similarity from given vector.")
    # group for input files
    group_input = argparser_caluculate.add_argument_group("Input files arguments")
    group_input.add_argument("-i", "--input", dest = "input_file", type = str, required = True,
                              help = "Input file must be csv files written vectorize image tags. REQUIRED.")
    group_input.add_argument("-t", "--title", dest = "title", type = str, required = True,
                              help = "named output file. REQUIRED.")
    # group for output files
    group_output = argparser_caluculate.add_argument_group("Output arguments")
    add_outdir_option(group_output)
    group_output.add_argument("-n", "--name", dest = "name", type = str,
                               help = "Project name, which will be used to generate output file names. DEFAULT: \"NA\"",
                               default = "NA")
def add_search_parser(subparsers):
    """Add main function 'search' argument parsers.
    """   
    argparser_search = subparsers.add_parser("search", help = "Search similar images from given image files by using cosine similarity.")
    # group for input files
    group_input = argparser_search.add_argument_group("Input files arguments")
    group_input.add_argument("-i", "--input", dest = "input_file", type = str, required = True,
                              help = "Input file must be csv files written cosine similarity and image files. REQUIRED.")
    group_input.add_argument("-t", "--title", dest = "title", type = str, required = True,
                              help = "named output file. REQUIRED.")
    # group for output files
    group_output = argparser_search.add_argument_group("Output arguments")
    add_outdir_option(group_output)
    group_output.add_argument("-n", "--name", dest = "name", type = str,
                               help = "Project name, which will be used to generate output file names. DEFAULT: \"NA\"",
                               default = "NA")
#def time():
#    import time
#    # 処理前の時刻(t0)を取得
#    t0 = time.clock()
#    # 計測したい処理
#    main()
#    # 処理後の時刻(t1)を取得
#    t1 = time.clock()
#    # 処理後の時刻(t1)-処理前の時刻(t0)で処理時間を計算
#    print("dt="+str(t1-t0)+"[s]")
# 
if __name__ == "__main__":
    try:
            main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupted me! ;-) Bye!\n")
        sys.exit(0)
