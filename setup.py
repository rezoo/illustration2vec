#!/usr/bin/env python
from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()


setup(
    name="illustration2vec",
    version="2.0.1",
    packages=find_packages(),
    install_requires=[
        'appdirs==1.4.3',
        'arrow>=0.12.1',
        'chainer>=4.1.0',
        'click>=6.7',
        'Flask-Admin==1.5.1',
        'Flask-Migrate==2.1.1',
        'flask-shell-ipython==0.3.0',
        'Flask-SQLAlchemy==2.3.2',
        'Flask-WTF==0.14.2',
        'Flask==1.0.2',
        'numpy>=1.14.3',
        'Pillow>=5.1.0',
        'scikit-image>=0.14.0',
        'SQLAlchemy-Utils>=0.33.3',
        'structlog==18.1.0',
    ],
    author="rezoo",
    author_email="rezoolab@gmail.com",
    maintainer="Rachmadani Haryono",
    maintainer_email="foreturiga@gmail.com",
    description="A simple deep learning library for estimating a set of tags "
    "and extracting semantic feature vectors from given illustrations.",
    license="MIT License",
    keywords="machine learning tag image illustration",
    url="https://github.com/rezoo/illustration2vec/",   # project home page, if any
    project_urls={
        "Bug Tracker": "https://github.com/rezoo/illustration2vec/issues",
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    zip_safe=True,
    entry_points={'console_scripts': ['i2v = i2v.__main__:cli', ],},
    extras_require={
        'dev':  [
            'pdbpp>=0.9.2',
            'ipython>=6.4.0',
        ],
    },
)
