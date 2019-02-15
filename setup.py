from setuptools import setup
from os import path

# Pull TLS version from single source of truth file
try:  # Python 2
    execfile(path.join("transitleastsquares", 'version.py'))
except:  # Python 3
    exec(open(path.join("transitleastsquares", 'version.py')).read())


# If Python3: Add "README.md" to setup. 
# Useful for PyPI (pip install transitleastsquares). Irrelevant for users using Python2
try:
    
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except:
    long_description = ' '

    
setup(name='transitleastsquares',
    version=TLS_VERSIONING,
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
        'astropy<3;python_version<"3"',  # astropy 3 doesn't install in Python 2, but is req for astroquery
        'astroquery>=0.3.9',  # earlier has bug for "from astroquery.mast import Catalogs"
        'numpy',
        'numba',
        'tqdm',
        'batman-package',
        'argparse',
        'configparser'
        ]
)
