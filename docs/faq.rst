.. currentmodule:: orthoseg

===
FAQ
===

.. _FAQ-standalone-scripts:

What model should I use?
------------------------

I first looked around to several sources to find the most promising neural networks to
use for image segmentation.

Interesting pages I found in this search are the following:

* A very interesting comparison of the performance of different DNNs:
  `Benchmark Analysis of Representative Deep Neural Network Architectures <https://arxiv.org/pdf/1810.00736.pdf>`_.
  Not specific to image segmentation though, rather object classification, but
  interesting to choose a good backbone DNN for the segmentation.
* This is an overview of the accuracy and network size of the different pretrained
  neural networks that are available in keras:
  `Keras models <https://keras.io/api/applications/>`_.
* paperswithcode.com is also an interesting website with an huge overview of accuracy
  results achieved using AI, also in the domain of computer vision. Some examples:

  * `Best performing image classifications on the imagenet dataset <https://paperswithcode.com/sota/image-classification-on-imagenet>`_
  * `Best performing image segmentations on the PASCAL VOC 2012 dataset <https://paperswithcode.com/sota/semantic-segmentation-on-pascal-voc-2012>`_

Based on the tests I did, inceptionresnetv2 gave me the best combination of quality and
performance as a backbone, so this is the default in orthoseg. The last time I did such
tests was in 2024.
