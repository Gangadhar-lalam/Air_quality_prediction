from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]
    return requirements

setup(
    name="Air_quality_prediction",
    version="0.0.1",
    author="gangadhar",
    author_email="s210267@rguktsklm.ac.in",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")

)