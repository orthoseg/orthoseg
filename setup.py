import os
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

version = "0.1.8"
os.environ['PACKAGE_TAG_VERSION'] = version

setuptools.setup(
    name="orthoseg", 
    version=version,
    author="Pieter Roggemans",
    author_email="pieter.roggemans@gmail.com",
    description="Package to make it easier to segment orthophotos.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theroggy/orthoseg",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
            "tensorflow>=2.2,<2.3", "pillow", "rasterio", "geopandas>=0.8,<0.9", 
            "owslib", "segmentation-models>=1.0,<1.1", "geofileops==0.0.7"],
    entry_points='''
            [console_scripts]
            orthoseg=orthoseg.orthoseg:main
            ''',
    classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)