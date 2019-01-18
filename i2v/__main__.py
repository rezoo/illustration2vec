#!/usr/bin/env python
from pprint import pprint
import hashlib

from PIL import Image
import click

from . import make_i2v_with_chainer

@click.group()
@click.version_option()
def cli():
    pass

@cli.command()
@click.option('--output', default='default')
@click.argument('images', nargs=-1)
def estimate_plausible_tags(images, output='default'):
    """Estimate plausible tags."""
    illust2vec = make_i2v_with_chainer(
        "illust2vec_tag_ver200.caffemodel", "tag_list.json")
    image_sets = map(lambda x: {'value': x, 'sha256':sha256_checksum(x)}, images)
    for idx, img_set in enumerate(image_sets):
        img = Image.open(img_set['value'])
        res = illust2vec.estimate_plausible_tags([img], threshold=0.5)
        if isinstance(res[idx]['rating'], zip):
            res[idx]['rating'] = list(res[idx]['rating'])
        print("image: {}\nsha256: {}".format(
            img_set['value'], img_set['sha256']))
        if output == 'pprint':
            pprint(res)
        else:
            print(res)


def sha256_checksum(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()


if __name__ == '__main__':
    cli()
