#!/usr/bin/env python3
"""Model module."""
from datetime import datetime
import hashlib
import os
import os.path as op

from appdirs import user_data_dir
from flask_admin import form
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.event import listens_for
from sqlalchemy.types import TIMESTAMP
from sqlalchemy_utils.types.choice import ChoiceType

MODE_PLAUSIBLE_TAG = 'plausible'
MODE_TOP_TAG = 'top'
db = SQLAlchemy()
file_path = op.join(user_data_dir('Illustration2Vec', 'Masaki Saito'), 'files')


class Base(db.Model):
    __abstract__ = True
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(TIMESTAMP, default=datetime.now, nullable=False)


class Image(Base):
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String)
    checksum_id = db.Column(db.Integer, db.ForeignKey('checksum.id'))
    checksum = db.relationship(
        'Checksum', foreign_keys='Image.checksum_id', lazy='subquery',
        backref=db.backref('images', lazy=True, cascade='delete'))

    def __repr__(self):
        return '<Image {0.id}>'.format(self)

    @property
    def full_path(self):
        return os.path.join(file_path, self.path)

    def update_checksum(self, session=None):
        session = db.session if session is None else session
        checksum_val = sha256_checksum(self.full_path)
        self.checksum = get_or_create(session, Checksum, value=checksum_val)[0]

    @property
    def thumbgen_filename(self):
        return form.thumbgen_filename(self.path)


def sha256_checksum(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()


@listens_for(Image, 'after_delete')
def del_image(mapper, connection, target):
    if target.path:
        # Delete image
        try:
            os.remove(op.join(file_path, target.path))
        except OSError:
            pass
        # Delete thumbnail
        try:
            os.remove(op.join(file_path, form.thumbgen_filename(target.path)))
        except OSError:
            pass


class Checksum(Base):
    value = db.Column(db.String, unique=True)

    def update_tag_estimation(self, tags, mode=MODE_PLAUSIBLE_TAG, session=None):
        session = db.session if session is None else session
        for nm, list_value in tags.items():
            if list_value:
                for tag_value, estimation_value in list_value:
                    tag_model = get_or_create_tag(
                        value=str(tag_value), namespace=nm, session=session)[0]
                    e_item = get_or_create(
                        session, TagEstimation,
                        checksum=self, tag=tag_model, mode=mode)[0]
                    e_item.value = estimation_value
                    yield e_item

    def __repr__(self):
        return '<Checksum {0.id} {0.value}>'.format(self)

    def get_estimated_tags(self, mode=MODE_PLAUSIBLE_TAG):
        res = {'character': [], 'copyright': [], 'general': [], 'rating': []}
        for estimation in self.tag_estimations:
            if estimation.mode == mode:
                res.setdefault(estimation.tag.namespace.value, []).append(
                    (estimation.tag.value, estimation.value))
        return res


class TagEstimation(Base):
    MODES = [
        (MODE_PLAUSIBLE_TAG, MODE_PLAUSIBLE_TAG),
        (MODE_TOP_TAG, MODE_TOP_TAG),
    ]
    checksum_id = db.Column(db.Integer, db.ForeignKey('checksum.id'))
    checksum = db.relationship(
        'Checksum', foreign_keys='TagEstimation.checksum_id', lazy='subquery',
        backref=db.backref('tag_estimations', lazy=True))
    tag_id = db.Column(db.Integer, db.ForeignKey('tag.id'))
    tag = db.relationship(
        'Tag', foreign_keys='TagEstimation.tag_id', lazy='subquery',
        backref=db.backref('tag_estimations', lazy=True))
    value = db.Column(db.Float)
    mode = db.Column(ChoiceType(MODES))

    def __repr__(self):
        templ = \
            '<TagEstimation {0.id} mode:{0.mode.value} {0.tag.fullname} confidence:{1}>'
        return templ.format(
            self, '{0:.2f}'.format(self.value * 100)
        )


def get_or_create_tag(value, namespace=None, session=None):
    session = db.session if session is None else session
    kwargs = dict(value=value)
    if namespace:
        namespace_model = get_or_create(session, Namespace, value=namespace)[0]
        kwargs['namespace'] = namespace_model
    model, created = get_or_create(session, Tag, **kwargs)
    return model, created


class Tag(Base):
    value = db.Column(db.String)
    namespace_id = db.Column(db.Integer, db.ForeignKey('namespace.id'))
    namespace = db.relationship(
        'Namespace', foreign_keys='Tag.namespace_id', lazy='subquery',
        backref=db.backref('tags', lazy=True))

    @property
    def fullname(self):
        res = ''
        if self.namespace:
            res = self.namespace.value + ':'
        res += self.value
        return res

    def __repr__(self):
        return '<Tag {0.id} {0.fullname}>'.format(self)


class Namespace(Base):
    value = db.Column(db.String, unique=True)


def get_or_create(session, model, **kwargs):
    """Creates an object or returns the object if exists."""
    instance = session.query(model).filter_by(**kwargs).first()
    created = False
    if not instance:
        instance = model(**kwargs)
        session.add(instance)
        created = True
    return instance, created
