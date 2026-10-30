from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the version from bagpipes/__init__.py without importing the package
with open(path.join(here, 'bagpipes', '__init__.py'), encoding='utf-8') as f:
    version = [l.split('"')[1] for l in f if l.startswith('__version__')][0]

setup(
    name='bagpipes',

    version=version,

    description='Galaxy spectral fitting',

    long_description=long_description,

    long_description_content_type='text/markdown',

    url='https://bagpipes.readthedocs.io',

    author='Adam Carnall, wfescAddition by P. Oesch',

    author_email='adamc@roe.ac.uk',

    packages=["bagpipes", "bagpipes.fitting", "bagpipes.catalogue",
              "bagpipes.models", "bagpipes.filters", "bagpipes.input",
              "bagpipes.plotting", "bagpipes.models.making", "bagpipes.moons"],

    include_package_data=True,

    install_requires=["numpy", "corner", "pymultinest>=2.11", "h5py", "pandas",
                      "astropy", "matplotlib>=2.2.2", "scipy", "msgpack",
                      "spectres", "nautilus-sampler>=1.0.2"],

    project_urls={
        "readthedocs": "https://bagpipes.readthedocs.io",
        "GitHub": "https://github.com/ACCarnall/bagpipes",
        "ArXiv": "https://arxiv.org/abs/1712.04452"
    }
)
