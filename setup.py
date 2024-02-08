from typing import Dict

from setuptools import find_packages, setup

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import snorkel.
VERSION: Dict[str, str] = {}
with open("text_guided_cl/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

with open('requirements.txt') as f:
    requirements = f.readlines()

test_requirements = ['pytest']

# Use README.md as the long_description for the package
with open("README.md", "r") as readme_file:
    long_description = readme_file.read()
print(requirements)
setup(
    name="text_guided_cl",
    version=VERSION["__version__"],
    description="Text-Guided Image Clustering project.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    author_email="",
    license="Apache License 2.0",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=requirements,
    tests_require=test_requirements,
    extras_require={'test': test_requirements}
)
