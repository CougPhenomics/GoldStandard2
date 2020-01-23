from plantcv import plantcv as pcv
import os
from matplotlib import pyplot as plt
import cv2

pcv.params.debug= 'plot'
plt.rcParams['figure.figsize'] = [12, 12]

visfn = "data/vis/A6-GoldStandard2_RGB-20190801T043947-VIS0-0.png"
psIIfn = "data/psII/A6-GoldStandard2_PSII-20190801T003230-PSII0-2.png"

visimg, bn, fn = pcv.readimage(visfn)
psIIimg, bn, fn = pcv.readimage(psIIfn)

vis_x = visimg.shape[1]
vis_y = visimg.shape[0]
psII_x = psIIimg.shape[1]
psII_y = psIIimg.shape[0]

masko = pcv.threshold.otsu(psIIimg,255, 'light')
mask = pcv.erode(masko,2,2)
final_mask = mask

mask_shift_x = pcv.shift_img(final_mask, 14, 'left')
mask_shift_y = pcv.shift_img(mask_shift_x, 3, 'top')

# vis_mask = pcv.resize(final_mask, resize_x = vis_x/psII_x, resize_y=vis_y/psII_y)

vis_mask2 = cv2.resize(mask_shift_y, (vis_x, vis_y),
                       interpolation=cv2.INTER_CUBIC)

vis_masked = pcv.apply_mask(visimg, vis_mask2, 'black')


# vs_ws=pcv.watershed_segmentation(visimg,vis_mask2)