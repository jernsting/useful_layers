try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

__version__ = '0.0.1'

setup(
    name='useful_layers',
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    description="""
Useful Layers
Useful Layers is a torch based library containing some experimental,
but useful layers
""",
    author='Jan Ernsting',
    author_email='j.ernsting@uni-muenster.de',
    url='https://github.com/jernsting/useful_layers.git',
    download_url='https://github.com/jernsting/useful_layers/archive/' +
    __version__ + '.tar.gz',
    keywords=['machine learning', 'deep learning', 'experimental', 'science'],
    classifiers=[],
)
