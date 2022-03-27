from setuptools import setup

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name = 'dptools',
      version = '0.4.1',
      description = 'Data Preprocessing Tools',
      long_description = long_description,
      long_description_content_type = 'text/markdown',
      author = 'Nikita Kozodoi',
      author_email = 'n.kozodoi@icloud.com',
      url = 'https://github.com/kozodoi/dptools',
      packages = ['dptools'],
      install_requires = ['numpy', 'pandas', 'scikit-learn', 'scipy'],
      license = 'MIT',
      zip_safe = False
     )