Imagelayers
===========

In this file, the image layers we can use are configured.


[BEFL-2019]
-----------

.. confval:: BEFL-2019.use_cache
   :type: ``str``
   :default: ``ifavailable``

   Use cache to store the downloaded images. If set to no, the images will be
   downloaded each time they are needed. Defaults to yes.
   Possible values: yes, no, ifavailable
   When set to ifavailable, the cache will be used if it exists, but if it does
   not exist, the image will be downloaded.

.. confval:: BEFL-2019.wms_server_url
   :type: ``str``
   :default: ``https://geo.api.vlaanderen.be/omw/wms?``

   WMS information of the layer

.. confval:: BEFL-2019.wms_layernames
   :type: ``str``
   :default: ``OMWRGB19VL``


.. confval:: BEFL-2019.wms_layerstyles
   :type: ``str``
   :default: ``default``


.. confval:: BEFL-2019.wms_version
   :type: ``str``
   :default: ``1.3.0``


.. confval:: BEFL-2019.projection
   :type: ``str``
   :default: ``epsg:31370``

   wms_username = XXX
   wms_password = YYY

.. confval:: BEFL-2019.switch_axes
   :type: ``str``
   :default: ``""``

   True if the x and y axes of the bounding boxes should be switched when reading data.
   This is needed for some projections, such as epsg:3059. If not specified, it will be
   determined automatically based on the projection.

.. confval:: BEFL-2019.bbox
   :type: ``str``
   :default: ``174900, 176400, 175300, 176600``

   Define the region of interest for this layer. When downloading images for
   the prediction cache, only images intersecting the ROI will be downloaded.
   Can be defined either as a bounding box or you can point to a geofile
   containing the ROI.
   Bbox of the ROI you want to use

.. confval:: BEFL-2019.grid_xmin
   :type: ``int``
   :default: ``0``

   bbox = 170000, 170000, 180000, 180000
   Path to a file containing the ROI you want to use
   roi_filepath = c:\geodata\vlaanderen.gpkg
   Location of the origin of the grid to use when downloading images for the
   prediction cache.

.. confval:: BEFL-2019.grid_ymin
   :type: ``int``
   :default: ``0``


.. confval:: BEFL-2019.nb_concurrent_calls
   :type: ``int``
   :default: ``1``

   Options to manage the load that will be generated on the WMS server.
   (Max) nb of parallel calls (defaults to 1).


[BEFL-TEST-s2-fields-2023]
--------------------------

Apply random nb of secs of sleep between 2 calls up to nb seconds specified.
random_sleep = 10
Remove a watermark on the border of the images.
If image_pixels_ignore_border is specified, an image of x pixels larger in
all directions is requested in the WMS call, but this border is removed again
when saving the image.
image_pixels_ignore_border = 100

.. confval:: BEFL-TEST-s2-fields-2023.use_cache
   :type: ``str``
   :default: ``no``

   Use cache to store the downloaded images. If set to no, the images will be
   downloaded each time they are needed. Defaults to yes.
   Possible values: yes, no, ifavailable
   When set to ifavailable, the cache will be used if it exists, but if it does
   not exist, the image will be downloaded.

.. confval:: BEFL-TEST-s2-fields-2023.path
   :type: ``str``
   :default: ``./fields/input_raster/BEFL-TEST-s2_2023-05-01_2023-07-01_B08-B04-B03_min_byte.tif``


.. confval:: BEFL-TEST-s2-fields-2023.projection
   :type: ``str``
   :default: ``epsg:32631``


.. confval:: BEFL-TEST-s2-fields-2023.layername
   :type: ``str``
   :default: ``BEFL-TEST-s2-fields-2023``


.. confval:: BEFL-TEST-s2-fields-2023.image_format
   :type: ``str``
   :default: ``image/png``

   Specify the format the images are requested and saved in. Defaults to image/jpeg.

.. confval:: BEFL-TEST-s2-fields-2023.bbox
   :type: ``str``
   :default: ``174900, 176200, 175300, 176700``

   Define the region of interest for this layer. When downloading images for
   the prediction cache, only images intersecting the ROI will be downloaded.
   Can be defined either as a bounding box or you can point to a geofile
   containing the ROI.
   Bbox of the ROI you want to use

.. confval:: BEFL-TEST-s2-fields-2023.grid_xmin
   :type: ``int``
   :default: ``0``

   bbox = 170000, 170000, 180000, 180000
   Path to a file containing the ROI you want to use
   roi_filepath = c:\geodata\vlaanderen.gpkg
   Location of the origin of the grid to use when downloading images for the
   prediction cache.

.. confval:: BEFL-TEST-s2-fields-2023.grid_ymin
   :type: ``int``
   :default: ``0``


.. confval:: BEFL-TEST-s2-fields-2023.nb_concurrent_calls
   :type: ``int``
   :default: ``2``

   Options to manage the load that will be generated on the WMS server.
   (Max) nb of parallel calls (defaults to 1).


[BEFL-2019-WMTS]
----------------

Apply random nb of secs of sleep between 2 calls up to nb seconds specified.
random_sleep = 10
Remove a watermark on the border of the images.
If image_pixels_ignore_border is specified, an image of x pixels larger in
all directions is requested in the WMS call, but this border is removed again
when saving the image.
image_pixels_ignore_border = 100

.. confval:: BEFL-2019-WMTS.wmts_server_url
   :type: ``str``
   :default: ``https://geo.api.vlaanderen.be/omw/wmts?``

   WMTS information of the layer

.. confval:: BEFL-2019-WMTS.wmts_layernames
   :type: ``str``
   :default: ``omwrgb19vl``


.. confval:: BEFL-2019-WMTS.wmts_layerstyles
   :type: ``str``
   :default: ``default``


.. confval:: BEFL-2019-WMTS.wmts_version
   :type: ``str``
   :default: ``1.0.0``


.. confval:: BEFL-2019-WMTS.wmts_tile_matrix_set
   :type: ``str``
   :default: ``BPL72VL``


.. confval:: BEFL-2019-WMTS.projection
   :type: ``str``
   :default: ``epsg:31370``


.. confval:: BEFL-2019-WMTS.layername
   :type: ``str``
   :default: ``BEFL-2019-WMTS``


.. confval:: BEFL-2019-WMTS.image_format
   :type: ``str``
   :default: ``image/jpeg``

   Specify the format the images are requested and saved in. Defaults to image/jpeg.

.. confval:: BEFL-2019-WMTS.bbox
   :type: ``str``
   :default: ``174900, 176400, 175300, 176600``

   Bbox of the ROI you want to use

.. confval:: BEFL-2019-WMTS.grid_xmin
   :type: ``int``
   :default: ``0``

   Location of the origin of the grid to use when downloading images for the
   prediction cache.

.. confval:: BEFL-2019-WMTS.grid_ymin
   :type: ``int``
   :default: ``0``



[OSM-XYZ]
---------

.. confval:: OSM-XYZ.path
   :type: ``str``
   :default: ``./imagelayer_osm.xml``

   The GDAL WMS driver can be used to use several types of services. An .xml file is used
   to configure the layer. This layer is an example showing how to configure an XYZ tile
   service. More information can be found here: https://gdal.org/drivers/raster/wms.html

.. confval:: OSM-XYZ.layername
   :type: ``str``
   :default: ``OSM_XYZ``


.. confval:: OSM-XYZ.projection
   :type: ``str``
   :default: ``epsg:3857``


.. confval:: OSM-XYZ.image_format
   :type: ``str``
   :default: ``image/jpeg``

   Specify the format the images are requested and saved in. Defaults to image/jpeg.

.. confval:: OSM-XYZ.bbox
   :type: ``str``
   :default: ``525500, 6603100, 525600, 6603200``

   Bbox of the ROI you want to use

.. confval:: OSM-XYZ.nb_concurrent_calls
   :type: ``int``
   :default: ``4``

   (Max) nb of parallel calls (optional)
