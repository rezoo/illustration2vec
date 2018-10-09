#!/usr/bin/env python
from pprint import pprint
import os
import os.path as op
import sys

from flask import Flask, __version__ as flask_version, send_from_directory
from flask.cli import FlaskGroup
from flask_admin import Admin
from PIL import Image
import click

from . import make_i2v_with_chainer, views, models


__version__ = '0.2.1'


class CustomFlaskGroup(FlaskGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.params[0].help = 'Show the program version'
        self.params[0].callback = get_custom_version


def get_custom_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    message = 'Illustration2Vec %(app_version)s\nFlask %(version)s\nPython %(python_version)s'
    click.echo(message % {
        'app_version': __version__,
        'version': flask_version,
        'python_version': sys.version,
    }, color=ctx.color)
    ctx.exit()


def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = 'True'
    app.config['DATABASE_FILE'] = 'i2v.sqlite'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + app.config['DATABASE_FILE']
    app.config['SECRET_KEY'] = os.getenv('ILLUSTRATION2VEC_SECRET_KEY') or os.urandom(24)
    # Create directory for file fields to use
    try:
        os.mkdir(models.file_path)
        app.logger.debug('File path created')
    except OSError:
        pass
    # app and db
    models.db.init_app(app)
    app.app_context().push()
    models.db.create_all()
    # other setup
    admin = Admin(
        app, name='Illustration2Vec', template_mode='bootstrap3',
        index_view=views.HomeView(url='/')
    )
    admin.add_view(views.ImageView(models.Image, models.db.session))
    admin.add_view(views.ChecksumView(models.Checksum, models.db.session))
    admin.add_view(views.TagEstimationView(models.TagEstimation, models.db.session))
    app.add_url_rule('/file/<filename>', 'file', view_func=lambda filename: send_from_directory(models.file_path, filename))
    app.logger.debug('file path: {}'.format(models.file_path))
    return app


@click.group(cls=CustomFlaskGroup, create_app=create_app)
def cli():
    """Illustration2Vec."""
    pass


@cli.command()
@click.option('--output', help='Output format;[default]/pprint', default='default')
@click.argument('images', nargs=-1)
def estimate_plausible_tags(images, output='default'):
    """Estimate plausible tags."""
    illust2vec = make_i2v_with_chainer(
        "illust2vec_tag_ver200.caffemodel", "tag_list.json")
    image_sets = map(lambda x: {'value': x, 'sha256': models.sha256_checksum(x)}, images)
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


if __name__ == '__main__':
    cli()
