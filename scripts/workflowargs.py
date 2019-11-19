import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import re
import numpy as np
import argparse
import cv2
from plantcv import plantcv as pcv
import os
from matplotlib import pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module='plotnine')

plt.rcParams['figure.figsize'] = [15,15]


class options():
    def __init__(self):
        self.image = "c:/users/dominikschneider/Documents/phenomics/GoldStandard2/data/testimages/vis/A3-GoldStandard2_RGB-20190719T161032-VIS0.png"
        self.outdir = "output/vistest"
        self.result = "output/vistest/result.json"
        self.regex = "(.{2})-(.+)-(\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})-(.+)"
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
