from setuptools import find_packages
from setuptools import setup

requirements = ["numpy"]

dev_requirements = [
    "pytest",
    "pytest-cov",
    "black",
    "pre-commit",
    "pre-commit-hooks",
    "flake8",
    "flake8-bugbear",
    "pep8-naming",
    "reorder-python-imports",
    "Pygments",
]

setup(
    name="robolearn",
    version="0.1.0",
    packages=find_packages(),
    author=["UWRobotLearningLab"],
    author_email=["rosario@cs.uw.edu"],
    url="http://github.com/UWRobotLearning/repo-template",
    install_requires=requirements,
    extras_require={"dev": dev_requirements},
)
