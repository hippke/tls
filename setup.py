import sys
from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))

# Pull TLS version from "config.cfg"
if sys.version_info[0] < 3:
    TLS_VERSION = '1'
else:
    from configparser import ConfigParser
    config = ConfigParser()
    config.read(path.join(this_directory, "transitleastsquares", "version.cfg"))
    TLS_VERSION = config["TLS"]["Version"]

# If Python3: Add "README.md" to setup. 
# Useful for PyPI (pip install transitleastsquares). Irrelevant for users using Python2
try:
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except:
    long_description = ' '

    
setup(name='transitleastsquares',
    version=TLS_VERSION,
    description='An optimized transit-fitting algorithm to search for periodic transits of small planets',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hippke/tls',
    author='Michael Hippke',
    author_email='michael@hippke.org',
    license='MIT',
    packages=['transitleastsquares'],
    include_package_data=True,
    package_data={'': ['*.csv', '*.cfg']},
    entry_points = {'console_scripts': ['transitleastsquares=transitleastsquares.command_line:main'],},
    install_requires=[
        'astroquery',
        'numpy',
        'numba',
        'tqdm',
        'batman-package',
        'argparse',
        'configparser'
        ]
)
