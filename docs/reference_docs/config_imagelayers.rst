.. _image-layers-configuration:

==========================
Image Layers Configuration
==========================

Image layers are configured in an ``imagelayers.ini`` file. Each INI section
defines one layer. The section name is used as the layer identifier.

There are three types of image sources supported by orthoseg:

- **WMS**: a standard OGC Web Map Service.
- **WMTS**: a standard OGC Web Map Tile Service.
- **File / GDAL**: a local raster file, a directory containing raster files,
  or any source supported by the GDAL WMS driver (including e.g. XYZ tile services
  configured via an ``.xml`` file).

The layer type is determined automatically based on which keys are
present in the section.

.. note::

   An example :doc:`/file_viewers/imagelayers_ini_viewer` is shipped with the sample projects so you
   can use it as a starting point.


WMS layers
----------

A section is treated as a WMS layer when ``wms_server_url`` is present.

.. confval:: wms_server_url

   :type: ``str``
   :required: yes
   :default: *(none)*

   Base URL of the WMS endpoint, e.g.
   ``https://geo.api.vlaanderen.be/omw/wms?``.

.. confval:: wms_layernames

   :type: ``str``
   :required: yes
   :default: *(none)*

   Comma-separated list of WMS layer names to request.

.. confval:: wms_layerstyles

   :type: ``str``
   :required: no
   :default: ``default``

   Comma-separated list of styles, one per layer listed in
   :confval:`wms_layernames`.

.. confval:: wms_version

   :type: ``str``
   :required: no
   :default: ``1.3.0``

   WMS protocol version, e.g. ``1.1.1`` or ``1.3.0``.

.. confval:: wms_username

   :type: ``str``
   :required: no
   :default: *(none)*

   Username for password-protected WMS services.

.. confval:: wms_password

   :type: ``str``
   :required: no
   :default: *(none)*

   Password for password-protected WMS services.

**Example** ::

   [BEFL-2019]
   wms_server_url = https://geo.api.vlaanderen.be/omw/wms?
   wms_layernames = OMWRGB19VL
   wms_layerstyles = default
   wms_version = 1.3.0
   projection = epsg:31370
   bbox = 174900, 176400, 175300, 176600
   use_cache = ifavailable
   nb_concurrent_calls = 1


WMTS layers
-----------

A section is treated as a WMTS layer when ``wmts_server_url`` is present.

.. confval:: wmts_server_url

   :type: ``str``
   :required: yes
   :default: *(none)*

   Base URL of the WMTS endpoint.

.. confval:: wmts_layernames

   :type: ``str``
   :required: yes
   :default: *(none)*

   Comma-separated list of WMTS layer names.

.. confval:: wmts_layerstyles

   :type: ``str``
   :required: no
   :default: ``default``

   Comma-separated list of styles.

.. confval:: wmts_version

   :type: ``str``
   :required: no
   :default: ``1.0.0``

   WMTS protocol version.

.. confval:: wmts_tile_matrix_set

   :type: ``str``
   :required: yes
   :default: *(none)*

   Tile matrix set identifier to use, e.g. ``BPL72VL``.

**Example** ::

   [BEFL-2019-WMTS]
   wmts_server_url = https://geo.api.vlaanderen.be/omw/wmts?
   wmts_layernames = omwrgb19vl
   wmts_layerstyles = default
   wmts_version = 1.0.0
   wmts_tile_matrix_set = BPL72VL
   projection = epsg:31370
   bbox = 174900, 176400, 175300, 176600


File and GDAL layers
--------------------

A section is treated as a file-based layer when the ``path`` key is present.

The path can point to:
   
- any georeference raster file, like a geotiff, a virtual raster (.vrt), or any other
  format that can be read by GDAL.
- a GDAL WMS ``.xml`` configuration file. This can be used to connect to web sources
  supported by GDAL, such as XYZ/TMS tile services.
- a directory containing georeferenced raster files that can be read by GDAL.

More details on these options are given below.

.. confval:: path

   :type: ``str``
   :required: yes
   :default: *(none)*

   Path to a local raster layer.
   
   This can be:
   
   - any georeference raster file, like a geotiff, a virtual raster (.vrt), or any other
     format that can be read by GDAL.
   - a GDAL WMS ``.xml`` configuration file. This can be used to connect to web sources
     supported by GDAL, such as XYZ/TMS tile services. See the
     `GDAL WMS driver documentation <https://gdal.org/drivers/raster/wms.html>`_ for
     details on how to write the ``.xml`` file.
   - a directory containing georeferenced raster files that can be read by GDAL.
     In this case, the ``file_patterns`` key must be used to specify which
     files belong to the layer.
     Orthoseg will create a virtual raster, ``orthoseg.vrt``, with all files that match
     the patterns specified in ``file_patterns`` and use it as the source for the layer.
     In addition, orthoseg will create a file called ``orthoseg.gpkg`` containing the
     footprints of all the individual files that match the patterns. This file will be
     used as the region of interest file (:confval:`roi_filepath`) for the layer.

.. confval:: file_patterns

   :type: ``str``
   :required: no
   :default: *(none)*

   Comma-separated list of glob patterns to select files when ``path`` points to a
   directory.

   Examples::

      # This selects all .tif in the directory, excluding subdirectories
      file_patterns = *.tif

      # This selects all .tif and .jpg files in the directory and all its subdirectories
      file_patterns = **/*.tif, **/*.jpg

**Example: local raster file** ::

   [BEFL-TEST-s2-fields-2023]
   path = ./fields/input_raster/BEFL-TEST-s2_2023-05-01_2023-07-01_B08-B04-B03_min_byte.tif
   projection = epsg:32631
   image_format = image/png
   bbox = 174900, 176200, 175300, 176700
   nb_concurrent_calls = 2

**Example: XYZ tile service via GDAL WMS XML** ::

   [OSM-XYZ]
   path = ./imagelayer_osm.xml
   projection = epsg:3857
   image_format = image/jpeg
   bbox = 525500, 6603100, 525600, 6603200
   nb_concurrent_calls = 4

See the `GDAL WMS driver documentation <https://gdal.org/drivers/raster/wms.html>`_ for
details on how to write the ``.xml`` file.

**Example: directory of raster files** ::

   [RANDOM-RASTER-FILES]
   path = ./random_raster_files/
   file_patterns = **/*.tif, **/*.jpg
   projection = epsg:32631
   image_format = image/jpeg


Common keys
-----------

The following keys can be used for all layer types.

.. confval:: use_cache

   :type: ``str``
   :required: no
   :default: ``yes``

   Whether to cache downloaded images on disk.

   Possible values:

   - ``yes``: always use the on-disk cache.
   - ``no``: never cache; re-download every time.
   - ``ifavailable``: use the cache when it exists, otherwise download.

.. confval:: projection

   :type: ``str``
   :required: yes
   :default: *(none)*

   The CRS of the layer expressed as an EPSG code, e.g. ``epsg:31370``.

.. confval:: switch_axes

   :type: ``bool``
   :required: no
   :default: *(auto-detected)*

   Set to ``True`` if the x and y axes of bounding boxes must be swapped when
   querying the service. When left empty orthoseg determines this
   automatically from the projection.

.. confval:: layername

   :type: ``str``
   :required: no
   :default: *(section name)*

   Override the layer name used internally. Defaults to the INI section name.

.. confval:: image_format

   :type: ``str``
   :required: no
   :default: ``image/jpeg``

   MIME type for the images that are requested and stored, e.g. ``image/jpeg``,
   ``image/png``.

.. confval:: bbox

   :type: ``str``
   :required: no
   :default: *(none)*

   Bounding box of the region of interest (ROI) for the layer.
   
   IT needs to be specified as four comma-separated coordinates in the layer's
   projection: ``xmin, ymin, xmax, ymax``.
   
   When creating a prediction cache or running a prediction without cache, only tiles
   of the prediction grid that intersect the ROI are downloaded/predicted.

   If your ROI is not a simple rectangle, you can use :confval:`roi_filepath` as an
   alternative, as this supports complex and/or multiple geometries.
   When both are specified, :confval:`bbox` takes precedence.

   Example::

      bbox = 174900, 176400, 175300, 176600

.. confval:: roi_filepath

   :type: ``str``
   :required: no
   :default: *(none)*

   Path to a vector file that defines the region of interest (ROI) for the layer.

   When creating a prediction cache or running a prediction without cache, only tiles
   of the prediction grid that intersect the ROI are downloaded/predicted.

   When the ROI is a simple rectangle, you can use :confval:`bbox` as an alternative.
   When both are specified, :confval:`bbox` takes precedence.

   Example::

      roi_filepath = ./roi_files/my_layer_roi.gpkg

.. confval:: grid_xmin

   :type: ``float``
   :required: no
   :default: ``0``

   X-coordinate of the grid origin used when tiling the prediction cache.

.. confval:: grid_ymin

   :type: ``float``
   :required: no
   :default: ``0``

   Y-coordinate of the grid origin used when tiling the prediction cache.

.. confval:: nb_concurrent_calls

   :type: ``int``
   :required: no
   :default: ``1``

   Maximum number of parallel requests sent to the image source.

.. confval:: random_sleep

   :type: ``float``
   :required: no
   :default: *(none)*

   When set, a random delay of up to this many seconds is used between consecutive
   requests to reduce server load.

.. confval:: image_pixels_ignore_border

   :type: ``int``
   :required: no
   :default: *(none)*

   When the image source adds a watermark or artefact on the border, set this
   to the number of border pixels to strip. orthoseg requests an image that
   is this many pixels larger in all directions and then crops the border
   before saving.
