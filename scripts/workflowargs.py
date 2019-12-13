import sys, traceback
import cv2
import os
import numpy as np
import argparse
import re
import string
import json
from plantcv import plantcv as pcv
from skimage import filters
from skimage import morphology
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module='plotnine')

plt.rcParams['figure.figsize'] = [15,15]


class options():
    def __init__(self):
        self.image = "data/vis/A6-GoldStandard2_RGB-20190726T161047-VIS0-0.png"
        self.outdir = "output/vistest"
        self.result = "output/vistest/result.json"
        self.regex = "(.{2})-(.+)-(\d{8}T\d{6})-(.+)-(\d{1})"
        self.debug = 'plot'
        self.debugdir = 'debug/vistest'


args = options()


def vismask(img):

    a_img = pcv.rgb2gray_lab(img, channel='a')
    thresh_a = pcv.threshold.binary(a_img, 124, 255, 'dark')
    b_img = pcv.rgb2gray_lab(img, channel='b')
    thresh_b = pcv.threshold.binary(b_img, 127, 255, 'light')

    mask = pcv.logical_and(thresh_a, thresh_b)
    mask = pcv.fill(mask, 800)
    final_mask = pcv.dilate(mask, 2, 1)

    return final_mask


def add_scalebar(pseudoimg, pixelresolution, barwidth, barlocation='lower center', fontprops=None, scalebar=None):
    if fontprops is None:
        fontprops = fm.FontProperties(size=16, weight='bold')

    ax = pseudoimg.gca()

    if scalebar is None:
        scalebar = AnchoredSizeBar(ax.transData,
                                   barwidth/pixelresolution,  '2 cm', barlocation,
                                   pad=0.5,
                                   sep=5,
                                   color='white',
                                   frameon=False,
                                   size_vertical=barwidth/pixelresolution/30,
                                   fontproperties=fontprops)

    ax.add_artist(scalebar)

    return ax.get_figure()
