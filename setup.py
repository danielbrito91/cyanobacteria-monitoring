from pathlib import Path
from setuptools import setup

BASE_DIR = Path(__file__).parent

with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name="src",
    version=0.1,
    description="Get cyanobacteria levels of a given lake",
    author="Daniel Oliveira de Brito",
    author_email="danieloliveiradebrito@gmail.com",
    python_requires=">=3.7",
    install_requires=[required_packages]
)