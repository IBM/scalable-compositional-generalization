from glob import glob
from setuptools import setup

setup(
    name="visgen",
    version="0.0",
    description="Visual Generalization study.",
    packages=["visgen"],
    data_files=[("configs", glob("configs/*.json"))],
    python_required=">=3.9",
)
