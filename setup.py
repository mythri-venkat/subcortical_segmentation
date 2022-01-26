from setuptools import setup, find_packages

exec(open('unetseg3d/__version__.py').read())
setup(
    name="unetseg3d",
    packages=find_packages(),
    version=__version__,
    author="Mythri V",
    python_requires='>=3.6.8'
)
