"""
Configuration setting to build the orthoseg package.
"""

import setuptools

with open("README.md") as fh:
    long_description = fh.read()

with open("orthoseg/version.txt") as file:
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
        "gdown",
        "geofileops>=0.6,<0.9",
        "geopandas>=0.12,<1",
        "owslib",
        "pillow",
        "pycron",
        "pygeoops>=0.2,<0.4",
        "rasterio",
        "segmentation-models>=1.0,<1.1",
        "shapely>=2",
        "simplification",
        "tensorflow>=2.7,<2.11",
    ],
    entry_points="""
            [console_scripts]
            orthoseg_load_images=orthoseg.load_images:main
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
    python_requires=">=3.8",
)
