# CHANGELOG

## 0.4.0 (???)

### Improvements

- Add option to use topologic simplify (#22)
- Add option to specify the minimum probability in the result (#23)
- Add option to ignore wms url in capabilities (#20)
- Add epsg 31468 as crs with switched xy axes (#18)
- Improve handling of empty features in training data (#16, #19)
- Add nb_parallel parameter to predict (#15)
- Apply black formatting to comply with pep8 (#27)

### Bugs fixed

- 

### Deprecations and compatibility notes

- 

## 0.3.0 (2022-02-10)

Highlights for this release are a feature to be able to combine bands from different WMS layers to one (3 band) input layer, performance improvements,...

### Improvements

- Add feature to combine bands from different WMS layers in one input image layer to segment
- Add option to save model after default 100 epochs even if min. accuracy not reached
- Add option to first train some epochs with the encode layers frozen
- Improve input data checks in prepare_traindatasets
- Introduce "scriptrunner", to have more flexibility when running orthoseg tasks combined with other scripts
- Improve/update documentation + sample project
- Improve prediction performance (#5)
- Improve logging
- Update dependencies, eg. tensorflow 2.8, geofileops 0.3, geopandas 0.10,...

### Bugs fixed

- When downloading a greyscale layer from a WMS, it was created incorrectly

## 0.2.1 (2021-03-15)

The most important improvement is that multiclass classification is now fully supported.

For typical segmentation projects the changes should be backwards compatible. If you have overridden one of these configuration options however, it is a good idea to review the documentation in project_defaults.ini to check if there are caveats to watch out for:

train.image_augmentations, train.mask_augmentations
train.classes

### Improvements

- Add support for multiclass classification: check out the train.classes parameter in project_defaults.ini
- Add option to use a cron schedule to configure when images can be downloaded: check
  out the download.cron_schedule parameter in project_defaults.ini
- Add option to simplify vector output on-the-fly during prediction to make output
  smaller: check out the predict.simplify_algorythm and predict.simplify_tolerance
  parameters in project_defaults.ini
- Make postprocessing steps to be done configurable in config files: check out the
  postprocess section in project_defaults.ini
- Vectorize predictions on-the-fly during prediction to evade need for large temporary storage
- Add end-to-end tests using a test project
- Add option to save augmented images during training for troubleshooting: check out
  the train.save_augmented_subdir parameter in project_defaults.ini
- Cleanup support for uninteresting models
- Improve logging, progress reporting, type annotations
- Update tensorflow dependency to 2.4

### Bugs fixed

- Fix error in algorithm to determine if label files have changed since last training

## 0.1.2 (2020-05-29)

First packaged version.

## 0.1.0 (2018-09-06)

First working version.
