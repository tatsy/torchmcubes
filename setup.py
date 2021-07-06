# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['torchmcubes']

package_data = \
{'': ['*']}

install_requires = \
['torch>=1.4.1']

setup_kwargs = {
    'name': 'torchmcubes',
    'version': '0.1.0',
    'description': 'PyTorch implementation of marching cubes',
    'long_description': None,
    'author': 'tatsy',
    'author_email': 'tatsy.mail@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
