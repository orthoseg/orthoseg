# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:49:26 2018

@author: Pieter Roggemans
"""

'''
import sys
path = "/home/dpakhom1/dense_crf_python/"
sys.path.append(path)
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary
import skimage.io as io
'''

import numpy as np
import skimage
import skimage.morphology       # Needs to be imported explicitly as it is a submodule
from scipy import ndimage
#import pydensecrf.densecrf as dcrf

def region_segmentation(predicted_mask,
                        thresshold_ok: float = 0.5):
    
    # ???
    elevation_map = skimage.filters.sobel(predicted_mask)
    
    # First apply some basic thressholds...
    markers = np.zeros_like(predicted_mask)
    markers[predicted_mask < thresshold_ok] = 1
    markers[predicted_mask >= thresshold_ok] = 2

    # Clean    
    segmentation = skimage.morphology.watershed(elevation_map, markers)   
    
    # Remove holes in the mask
    segmentation = ndimage.binary_fill_holes(segmentation - 1)
    
    return segmentation

def thresshold(mask, thresshold_ok: float = 0.6):
    mask[mask >= thresshold_ok] = 1
    mask[mask < thresshold_ok] = 0
    
    return mask

'''
Dense crf didn't seem to work very good... so dropped it.

Usage eg.:
    #    image_pred = pp.dense_crf(image=image_arr, mask=image_pred)
    #    image_pred = (image_pred * 255).astype(np.uint8)
    #    image_pred = pp.thresshold(image_pred)
    #    image_pred = np.delete(image_pred, obj=0, axis=2)
    
def dense_crf(image, mask):
    height = mask.shape[0]
    width = mask.shape[1]

    mask = np.expand_dims(mask, 0)
    mask = np.append(1 - mask, mask, axis=0)

    d = dcrf.DenseCRF2D(width, height, 2)
    U = -np.log(mask)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(image, dtype=np.uint8)

    d.setUnaryEnergy(U)

#    d.addPairwiseGaussian(sxy=20, compat=3)
#    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=3, srgb=3, rgbim=img, compat=3)

    Q = d.inference(2)
    Q = np.argmax(np.array(Q), axis=0).reshape((height, width))

    return np.array(Q)
'''
'''
def postprocess():

    image = train_image

    softmax = final_probabilities.squeeze()

    softmax = processed_probabilities.transpose((2, 0, 1))

    # The input should be the negative of the logarithm of probability values
    # Look up the definition of the softmax_to_unary for more information
    unary = softmax_to_unary(processed_probabilities)

    # The inputs should be C-continious -- we are using Cython wrapper
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF(image.shape[0] * image.shape[1], 2)

    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                       img=image, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                         kernel=dcrf.DIAG_KERNEL,
                         normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)

    res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

    cmap = plt.get_cmap('bwr')

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.imshow(res, vmax=1.5, vmin=-0.4, cmap=cmap)
    ax1.set_title('Segmentation with CRF post-processing')
    probability_graph = ax2.imshow(np.dstack((train_annotation,)*3)*100)
    ax2.set_title('Ground-Truth Annotation')
    plt.show()
'''