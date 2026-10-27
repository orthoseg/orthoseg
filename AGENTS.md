# Agent instructions for Orthoseg

## Architecture

- Orthoseg is a configuration-driven Python package for geospatial image
  segmentation. The public Python API is re-exported from `orthoseg/__init__.py`;
  installed CLI entry points are defined in `setup.py`.
- A typical project pipeline is: load imagery into a tiled cache
  (`orthoseg_load_images`), validate/prepare train-validation-test datasets
  (`orthoseg_validate` / `orthoseg_train`), train a Keras model, predict over cached
  tiles into vector output, then run project-wide postprocessing. The workflow and
  CLI are documented in `docs/reference_docs/orthoseg_cli.rst`.
- Project `.ini` files select model, training, prediction, and postprocessing
  settings. Shared image-layer `.ini` files describe reusable imagery sources and
  spatial metadata. The configuration loader is
  `orthoseg/helpers/config_helper.py`; CLI overrides use `section.key=value`.
- `orthoseg/model/` contains model construction and model/weight helpers.
  `orthoseg/lib/` contains the training, prediction, dataset preparation, and
  postprocessing implementations; these are internal APIs and may change without
  notice. Prefer the top-level API or CLI for user-facing integrations.

## Build, test, and lint

- Create the development environment with `conda env create -f env-tf-dev.yml`;
  activate it with `conda activate orthoseg-tf-dev`.
- Run the test suite with `pytest tests/`. Run one test with, for example:
  `pytest tests/test_trainer.py::test_train_error_model_exists_no_preload`.
- Run the configured formatter, linter, and type checker with
  `pre-commit run --all-files`. Install the hooks for automatic checks with
  `pre-commit install`.
- Build the Sphinx HTML documentation with `make -C docs html`.
- The package declares Python `>=3.10`; CI currently tests Python 3.10, 3.11, and 3.12.

## Repository conventions

- In Python modules, define public functions before private helper functions.
- Keep successful behavior and error cases in separate tests. Use
  `pytest.mark.parametrize` when several cases share setup and assertions, keep
  one-off test names concise, and order test functions alphabetically within each
  test module.
- Keep project-specific behavior in configuration rather than adding per-project
  Python code. Load and validate project settings through
  `orthoseg.helpers.config_helper`; use its typed accessors for values such as paths,
  booleans, and numbers.
- Keep shared image-layer configuration separate from project configuration. Image
  acquisition and prediction depend on configured projections, bounds, tiling, and
  overlap; preserve those settings through cache and inference changes.
- Model architecture identifiers use an `encoder+decoder` form, such as
  `mobilenetv2+unet`; model construction and compatibility checks belong in
  `orthoseg/model/model_factory.py`.
- Keep user-facing functions in the top-level API/CLI modules; treat direct calls
  into `orthoseg.lib` as internal unless the public API documentation is
  deliberately updated.
