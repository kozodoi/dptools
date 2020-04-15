from setuptools import setup

setup(name = 'dptools',
      version = '0.1.0',
      description = 'Data Preprocessing Tools',
      author = 'Nikita Kozodoi',
      author_email = 'n.kozodoi@icloud.com',
      url = 'https://github.com/kozodoi/dptools',
      packages = ['dptools'],
      install_requires = ['numpy', 'pandas', 'sklearn'],
      license = 'MIT',
      zip_safe = False
     )