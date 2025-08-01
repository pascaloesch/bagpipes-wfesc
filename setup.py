from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='bagpipes',

    version='1.2.0+wfesc',

    description='Galaxy spectral fitting',

    long_description=long_description,

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
