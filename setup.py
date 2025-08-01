"""Configuration setting to build the orthoseg package."""

from pathlib import Path

import setuptools

with Path("README.md").open() as fh:
    long_description = fh.read()

with Path("orthoseg/version.txt").open() as file:
    version = file.readline()

setuptools.setup(
    name="orthoseg",
    version=version,
    author="Pieter Roggemans",
    author_email="pieter.roggemans@gmail.com",
    description="Package to make it easier to segment orthophotos.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/orthoseg/orthoseg",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "gdal",
        "gdown",
        "geofileops>=0.10",
        "geopandas>=1.0",
        "matplotlib",
        "numpy",
        "owslib",
        "pillow",
        "pycron",
        "pygeoops>=0.4",
        "rasterio",
        "segmentation-models>=1.0,<1.1",
        "shapely>=2",
        "simplification",
        "tensorflow>=2.8",
    ],
    entry_points="""
            [console_scripts]
            orthoseg_load_images=orthoseg.load_images:main
            orthoseg_validate=orthoseg.validate:main
            orthoseg_train=orthoseg.train:main
            orthoseg_predict=orthoseg.predict:main
            orthoseg_postprocess=orthoseg.postprocess:main
            osscriptrunner=orthoseg.scriptrunner:main
            orthoseg_load_sampleprojects=orthoseg.load_sampleprojects:main
            """,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
