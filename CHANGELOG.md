# CHANGELOG

## 0.5.0 (???)

### Improvements

- Improve performance of prepare_traindatasets for large label files (#116)
- Add check that train and validation data is mandatory (#118)
- Support geopandas 0.13 + remove dependency on pygeos (#121)
- Several small improvements to logging, linting,... (#110, #112, #122,...)

### Bugs fixed

- Fix occasional "database locked" error while predicting (#120)

## 0.4.2 (2023-04-19)

### Improvements

- Add support for username/password logon for WMS (#97)
- Improve performance of load_images (#99)

### Bugs fixed

- Fix download of model in load_sampleprojects (#100)

## 0.4.1 (2023-01-20)

### Improvements

- Avoid warning blockysize is not supported for jpeg when downloading images (#77)
- Add __version__ attribute to orthoseg (#80)
- Add option to specify reclassify to neighbour query as postprocessing (#86, #88)
- Reuse sample_project for tests so it is tested as well (#81)
- Improve test coverage (#79, #82, #83)
- Add support for shapely 2 (#92)

### Bugs fixed

- Fix occasional errors when predicting with topologic simplify (#69)
- Fix non blocking errors in prediction being ignored/not reported + format error email in html (#72, #76)
- Fix error in prediction on ubuntu 22.04 (#73)
- Fix error when prediction output dir doesn't exist yet (#75)
- Fix reclassify_to_neighbours giving undetermined result when 2+ neighbours are reclassification candidates (#84)

### Deprecations and compatibility notes

- Remove old model architectures: standard+unet and ternaus+unet (#82)

## 0.4.0 (2022-11-14)

### Improvements

- Add option to use topologic simplify (#22)
- Add option to specify the minimum probability in the result (#23)
- Add option to reclassify (e.g. small) features to the class of its neighbour (#30, #36)
- Add check if location BBOXs are of the right size (#38)
- Improve handling of empty features in training data (#16, #19)
- In the polygons label file, change default class column name to "classname" (#39)
- Add traindata type todo (#37)
- Add support for training data from layers in different projections (#56, #60)
- Add list of invalid geometries + coordinates in train mail (#40)
- Add option to ignore wms url in capabilities (#20)
- Add support for crs's with switched axes order (#50)
- Add nb_parallel parameter to predict (#15)
- Add pre-simplify with RDP when using (inline) LANG simplification (#29)
- Add option to disable ssl verification when downloading sample projects (#64)
- Apply black formatting to comply with pep8 (#27)
- Enable running CI tests using github actions (#42, #67)
- Support newer versions of used packages (#59, #61, #62)

### Deprecations and compatibility notes

- Disable default simplify in postprocess (#32)
- Command 'scriptrunner' renamed to 'osscriptrunner' (#59)

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
