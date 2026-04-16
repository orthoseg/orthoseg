Image Layers Configuration
==========================

Image layers are configured in an ``imagelayers.ini`` file. Each INI section
defines one layer; the section name is used as the layer identifier.

There are three types of image sources supported by orthoseg:

- **WMS**: a standard OGC Web Map Service.
- **WMTS**: a standard OGC Web Map Tile Service.
- **File / GDAL**: a local raster file or any source supported by the GDAL
  WMS driver (including XYZ tile services configured via an ``.xml`` file).

The layer type is determined automatically based on which parameters are
present in the section.

.. note::

   An example `imagelayers.ini`_ is shipped with the sample projects so you
   can use it as a starting point.

.. _imagelayers.ini: docs/_static/config_files/imagelayers.ini


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

A section is treated as a file-based layer when ``path`` is present. This
covers both local raster files (GeoTIFF, etc.) and any source supported by the
GDAL WMS driver, which is configured via an ``.xml`` file. The latter lets
you connect to XYZ/TMS tile services and other GDAL-supported web sources.

.. confval:: path

   :type: ``str``
   :required: yes
   :default: *(none)*

   Path to a local raster file or to a GDAL WMS ``.xml`` configuration file.

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

See the `GDAL WMS driver documentation
<https://gdal.org/drivers/raster/wms.html>`_ for details on how to write the
``.xml`` file.


Common parameters
-----------------

The following parameters apply to all layer types.

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

   MIME type for the images that are requested and stored, e.g.
   ``image/png``.

.. confval:: bbox

   :type: ``str``
   :required: no
   :default: *(none)*

   Bounding box of the region of interest as four comma-separated coordinates
   in the layer's projection: ``xmin, ymin, xmax, ymax``. Only images that
   intersect the ROI are downloaded when building a prediction cache.

   Example::

      bbox = 174900, 176400, 175300, 176600

.. confval:: roi_filepath

   :type: ``str``
   :required: no
   :default: *(none)*

   Path to a vector file whose geometry defines the region of interest.
   Use this as an alternative to :confval:`bbox` when you need an irregular
   boundary.

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

   When set, a random delay of up to this many seconds is inserted between
   consecutive requests to reduce server load.

.. confval:: image_pixels_ignore_border

   :type: ``int``
   :required: no
   :default: *(none)*

   When the image source adds a watermark or artefact on the border, set this
   to the number of border pixels to strip. orthoseg requests an image that
   is this many pixels larger in all directions and then crops the border
   before saving.
