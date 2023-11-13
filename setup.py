from setuptools import setup

requirements = [
    "numpy>=1.16",
    "torch>=2.0.1",
    "scipy>=1.3",
    "matplotlib>=3.7",
]

setup(name="mygp", version="1.0", install_requires=requirements, packages=["mygp"])
