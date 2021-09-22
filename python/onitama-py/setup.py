import setuptools
from pathlib import Path


with open('requirements.txt') as f:
    required = [f for f in f.read().splitlines() if not f.startswith("#")]

setuptools.setup(
    name='onitama',
    author="OGT",
    version='0.0.1',
    description="Onitama RL",
    long_description=Path("../../README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=required,
    python_requires='>=3.6'
)