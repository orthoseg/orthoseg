.. currentmodule:: orthoseg

===
FAQ
===

.. _FAQ-standalone-scripts:

What model should I use?
------------------------

For image segmentation, often an encoder-decoder architecture is used, where the encoder
is a pretrained DNN that extracts features from the input image, and the decoder
reconstructs the segmentation map from these features.

Both the encoder and decoder can be implemented using different neural network
architectures.

Decoder
^^^^^^^

For the decoder, following model architectures are supported:
- `UNet`
- `LinkNet`
- `FPN`
- `PSPNet`

Based on practical tests, `UNet` and `LinkNet` give the best results. Both in terms of
accuracy and inference speed, they are very similar. There are many variants on e.g.
`UNet` as well, but it seems that most give marginal accuracy improvements for
significant increases in complexity resulting in worse inference/train speed. Hence,
the default in orthoseg is the classic `UNet`.

Encoder
^^^^^^^

For the encoder there are even more options. Based on the information found and some
practical tests, `InceptionResnetV2` gives the best combination of quality and
performance as a backbone to segment orthophotos, so this is the default in orthoseg.

In the table below some models that were considered are listed and compared to
`InceptionResnetV2`:

* **Model**: the name of the model
* **Acc@1**: the top 1 classification accuracy on imagenet
* **Speed**: an indication on the inference speed, normalized on `InceptionResnetV2`
  having speed = 100
* **Support**: whether the model is supported in orthoseg
* **Remarks**: some remarks on the model, e.g. if practical tests have been conducted on
  segmentation performance,...

================== ===== ===== ======= =======
Model              Acc@1 Speed Support Remarks
================== ===== ===== ======= =======
vgg11              69.15   -           Tested* (=TernausNet): not very accurate
vgg16              70.79   46    Yes   accuracy ~ vgg11
vgg19              70.89   46    Yes   "
resnet18           68.24   29          "
resnet34           72.17   32          "
resnet50           74.81   41    Yes   
resnet101          76.58   60    Yes   
resnet152          76.66   77    Yes   
resnet50v2         69.73   36    Yes   
resnet101v2        71.93   53    Yes   accuracy ~ vgg11
resnet152v2        72.29   75    Yes   "
resnext50          77.36   69          
resnext101         78.48  110          slower + worse accuracy
densenet121        74.67   51    Yes   
densenet169        75.85   62    Yes   
densenet201        77.13   77    Yes   
inceptionv3        77.55   71    Yes   
xception           78.87   77          
inceptionresnetv2  80.03  100    Yes   good accuracy, speed OK
seresnet18         69.41   37          accuracy ~ vgg11
seresnet34         72.60   41          accuracy ~ vgg11
seresnet50         76.44   43            
seresnet101        77.92   59            
seresnet152        78.34   87            
seresnext50        78.74   70            
seresnext101       79.88  115          slower + worse accuracy
senet154           81.06  251          a lot slower
nasnetlarge        82.12  213          a lot slower
nasnetmobile       74.04   51             
mobilenet          70.36   28    Yes   accuracy ~ vgg11
mobilenetv2        71.63   33    Yes   accuracy ~ vgg11
EfficientNetB0     77.1    49          v2 is documented to be faster and more accurate
EfficientNetB1     79.1    56          "
EfficientNetB2     80.1    65          "   
EfficientNetB3     81.6    88          "   
EfficientNetB4     82.9   151          "   
EfficientNetB5     83.6   253          "
EfficientNetB6	   84.0   404          "
EfficientNetB7	   84.3   616          "
EfficientNetV2B0	 78.7	   -        
EfficientNetV2B1	 79.8	   -        
EfficientNetV2B2	 80.5	   -       
EfficientNetV2B3	 82.0	   -       
EfficientNetV2S    83.9	   -       
EfficientNetV2M    85.3    96    Yes   Tested*: similar theoretical accuracy, worse in practice
EfficientNetV2L    85.7	   -       
================== ===== ===== ======= =======

Here are some more details about some tests performed:

TernausNet + Unet
-----------------
* Both standard `Unet` and `TernausNet` (`Unet` with `vgg11` backbone) were tested on
  the detection of greenhouses. The accuracy of both was very similar and was decent,
  but clearly worse than `InceptionResNetV2`.
* Support for these models was removed from orthoseg to avoid having to upgrade them to
  stay supported with newer keras versions.

EfficientNetV2M
---------------
* The classification accuracy on imagenet is significantly better than
  `InceptionResNetV2` (85.3% vs 80.3% top 5 accuracy)
* The number of weights is similar, and train/inference speed is reported to be a lot
  faster than v1 of the EfficientNet family.
* In practice, the train/inference speed was almost the same as `InceptionResNetV2`, and
  the accuracy was significantly worse. The IOU score obtained on the training data was
  around 0.95, which is significantly worse than the 0.98 obtained with
  `InceptionResNetV2`. When running an actual detection and reviewing the results
  on-screen, it seeemed like types of sealed areas that were less represented in the
  training data and were narrow were detected significantly worse than was the case with
  `InceptionResNetV2`. Possbibly this could be solved by adding extra training data of
  this type, but as the IOU score improvement was very low as well, it isn't an
  improvement over `InceptionResNetV2`.
* The IOU score from the training obtained on a "sealed surfaces" detection was slightly
  higher than `InceptionResNetV2` (0.9791 vs 0.9765) but not significantly so. When
  running an actual detection and reviewing the results on-screen, it seeemed like types
  of sealed areas that were less represented in the training data and were narrow were
  detected significantly worse than was the case with `InceptionResNetV2`. Possbibly
  this can be solved by adding extra training data of this type, but as the IOU score
  improvement was very low as well, it doesn't seem to be an improvement compared to
  `InceptionResNetV2`.


These are some interesting sources of information:

* A very interesting comparison of the performance of different DNNs:
  `Benchmark Analysis of Representative Deep Neural Network Architectures <https://arxiv.org/pdf/1810.00736.pdf>`_.
  Not specific to image segmentation though, rather object classification, but
  interesting to choose a good backbone DNN for the segmentation.
* This is an overview of the accuracy, network size and inference speed cost of the
  different pretrained neural networks available in keras.applications:
  `Keras models <https://keras.io/api/applications/>`_.
* A similar overview with some other models:
  `Classification Models <https://github.com/qubvel/classification_models?tab=readme-ov-file#specification>`_
* paperswithcode.com is also an interesting website with an huge overview of accuracy
  results achieved using AI, also in the domain of computer vision. Some examples:

  * `Best performing image classifications on the imagenet dataset <https://paperswithcode.com/sota/image-classification-on-imagenet>`_
  * `Best performing image segmentations on the PASCAL VOC 2012 dataset <https://paperswithcode.com/sota/semantic-segmentation-on-pascal-voc-2012>`_
