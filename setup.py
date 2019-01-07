from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='transitleastsquares',
      version='1.0.9',
      description='An optimized transit-fitting algorithm to search for periodic transits of small planets',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/hippke/tls',
      author='Michael Hippke',
      author_email='michael@hippke.org',
      license='MIT',
      packages=['transitleastsquares'],
      include_package_data=True,
      package_data={
      '': ['*.csv']
      },
      install_requires=[
          'numpy',
          'scipy',
          'numba',
          'tqdm',
          'batman-package',
          'argparse',
          'kplr',
          'k2plr'
      ]
)