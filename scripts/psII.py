# %% Setup
# Export .png to outdir from LemnaBase using LT-db_extractor.py
from plantcv import plantcv as pcv
import importlib
import os
from datetime import datetime, timedelta
import cv2 as cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings
import importlib
from tinydb import TinyDB, Query

warnings.filterwarnings("ignore", module="matplotlib")
warnings.filterwarnings("ignore", module='plotnine')

# %% Import functions from src/ directory to get snaphots, create masks, and setup image classification
from src.data import import_snapshots
from src.segmentation import createmasks
from src.util import masked_stats
from src.viz import add_scalebar, custom_colormaps

# %% io directories
indir = os.path.join('data', 'psII')
# snapshotdir = indir
outdir = os.path.join('output', 'psII')
debugdir = os.path.join('debug', 'psII')
maskdir = os.path.join(outdir, 'masks')
fluordir = os.path.join(outdir, 'fluorescence')
os.makedirs(outdir, exist_ok=True)
os.makedirs(debugdir, exist_ok=True)
os.makedirs(maskdir, exist_ok=True)
os.makedirs(fluordir, exist_ok=True)
heterodb = TinyDB('output/psII/psII_heterogeneity.json')
# %% pixel pixel_resolution
# mm (this is approx and should only be used for scalebar)
pixelresolution = 0.2

# %% Import tif file information based on the filenames. If extract_frames=True it will save each frame form the multiframe TIF to a separate file in data/pimframes/ with a numeric suffix
fdf = import_snapshots.import_snapshots(indir, camera='psii')
fdf = fdf.query('jobdate > "2019-07-17"')#issues with psII camera before 7-17-2019
# %% Define the frames from the PSII measurements and merge this information with the filename information
pimframes = pd.read_csv(os.path.join('data', 'pimframes_map.csv'),
                        skipinitialspace=True)
# this eliminate weird whitespace around any of the character fields
fdf_dark = (pd.merge(fdf.reset_index(), pimframes, on=['frameid'],
                     how='right'))

# %% remove absorptivity measurements which are blank images
# also remove Ft_FRon measurements. THere is no Far Red light.
df = (fdf_dark.query(
    '~parameter.str.contains("Abs") and ~parameter.str.contains("FRon")',
    engine='python'))

# %% remove the duplicate Fm and Fo frames where frame = Fmp and Fp from frameid 5,6
df = (df.query(
    '(parameter!="FvFm") or (parameter=="FvFm" and (frame=="Fo" or frame=="Fm") )'
))

# %% Arrange dataframe of metadata so Fv/Fm comes first
param_order = pimframes.parameter.unique()
df['parameter'] = pd.Categorical(df.parameter,
                                 categories=param_order,
                                 ordered=True)
# %% Setup Debug parmaeters
# pcv.params.debug can be 'plot', 'print', or 'None'. 'plot' is useful if you are testing your pipeline over a few samples so you can see each step.
pcv.params.debug = 'plot'  # 'print' #'plot', 'None'
# Figures will show 9x9inches which fits my monitor well.
plt.rcParams["figure.figsize"] = (9, 9)

# plt.rcParams["font.family"] = "Arial"  # All text is Arial
# %% The main analysis function
# I like to reload my mask function to make sure it's the latest if I've been optimizing it
importlib.reload(createmasks)

# This function takes a dataframe of metadata that was created above. We loop through each pair of images to compute photosynthetic parameters
def image_avg(fundf):
    # Predefine some variables
    global c, h, roi_c, roi_h

    # Get the filename for minimum and maximum fluoresence
    fn_min = fundf.query('frame == "Fo" or frame == "Fp"').filename.values[0]
    fn_max = fundf.query('frame == "Fm" or frame == "Fmp"').filename.values[0]

    # Get the parameter name that links these 2 frames
    param_name = fundf['parameter'].iloc[0]

    # Create a new output filename that combines existing filename with parameter
    outfn = os.path.splitext(os.path.basename(fn_max))[0]
    outfn_split = outfn.split('-')
    # outfn_split[2] = datetime.strptime(fundf.jobdate.values[0],'%Y-%m-%d').strftime('%Y%m%d')
    outfn_split[2] = fundf.jobdate.dt.strftime('%Y%m%d').values[0]
    basefn = "-".join(outfn_split[0:-1])
    outfn_split[-1] = param_name
    outfn = "-".join(outfn_split)
    print(outfn)

    # Make some directories based on sample id to keep output organized
    plantbarcode = outfn_split[0]
    fmaxdir = os.path.join(fluordir, plantbarcode)
    os.makedirs(fmaxdir, exist_ok=True)

    # If debug mode is 'print', create a specific debug dir for each pim file
    if pcv.params.debug == 'print':
        debug_outdir = os.path.join(debugdir, outfn)
        os.makedirs(debug_outdir, exist_ok=True)
        pcv.params.debug_outdir = debug_outdir

    # read images and create mask from max fluorescence
    # read image as is. only gray values in PSII images
    imgmin, _, _ = pcv.readimage(fn_min)
    img, _, _ = pcv.readimage(fn_max)
    fdark = np.zeros_like(img)
    out_flt = fdark.astype('float32')  # <- needs to be float32 for imwrite

    if param_name == 'FvFm':
        # create mask
        mask = createmasks.psIImask(img)
        roi_c, roi_h = pcv.roi.multi(img,
                                    coord=(250, 200),
                                    radius=80,
                                    spacing=(0, 220),
                                    ncols=1,
                                    nrows=2)

        if len(np.unique(mask)) == 1:
            c = []
            YII = mask
            NPQ = mask
            newmask = mask
        else:
            # find objects and setup roi
            c, h = pcv.find_objects(img, mask)

            # setup individual roi plant masks
            newmask = np.zeros_like(mask)

            # compute fv/fm and save to file
            Fv, hist_fvfm = pcv.fluor_fvfm(fdark=fdark,
                                        fmin=imgmin,
                                        fmax=img,
                                        mask=mask,
                                        bins=128)
            YII = np.divide(Fv,
                            img,
                            out=out_flt.copy(),
                            where=np.logical_and(mask > 0, img > 0))

            # NPQ is 0
            NPQ = np.zeros_like(YII)

        cv2.imwrite(os.path.join(fmaxdir, outfn + '-fvfm.tif'), YII)
        # print Fm
        cv2.imwrite(os.path.join(fmaxdir, outfn + '-fmax.tif'), img)
        # NPQ will always be an array of 0s

    else:  # compute YII and NPQ if parameter is other than FvFm
        # use cv2 to read image becase pcv.readimage will save as input_image.png overwriting img
        newmask = cv2.imread(os.path.join(maskdir, basefn + '-FvFm-mask.png'),-1)
        if len(np.unique(newmask)) == 1:
            YII = np.zeros_like(newmask)
            NPQ = np.zeros_like(newmask)

        else:
            # compute YII
            Fvp, hist_yii = pcv.fluor_fvfm(fdark,
                                        fmin=imgmin,
                                        fmax=img,
                                        mask=newmask,
                                        bins=128)
            # make sure to initialize with out=. using where= provides random values at False pixels. you will get a strange result. newmask comes from Fm instead of Fm' so they can be different
            #newmask<0, img>0 = FALSE: not part of plant but fluorescence detected.
            #newmask>0, img<=0 = FALSE: part of plant in Fm but no fluorescence detected <- this is likely the culprit because pcv.apply_mask doesn't always solve issue.
            YII = np.divide(Fvp,
                            img,
                            out=out_flt.copy(),
                            where=np.logical_and(newmask > 0, img > 0))

            # compute NPQ
            Fm = cv2.imread(os.path.join(fmaxdir, basefn + '-FvFm-fmax.tif'), -1)
            NPQ = np.divide(Fm,
                            img,
                            out=out_flt.copy(),
                            where=np.logical_and(newmask > 0, img > 0))
            NPQ = np.subtract(NPQ,
                            1,
                            out=out_flt.copy(),
                            where=np.logical_and(NPQ >= 1, newmask > 0))

        cv2.imwrite(os.path.join(fmaxdir, outfn + '-yii.tif'), YII)
        cv2.imwrite(os.path.join(fmaxdir, outfn + '-npq.tif'), NPQ)

    # end if-else Fv/Fm

    # Make as many copies of incoming dataframe as there are ROIs so all results can be saved
    outdf = fundf.copy()
    for i in range(0, len(roi_c) - 1):
        outdf = outdf.append(fundf)
    outdf.frameid = outdf.frameid.astype('uint8')

    # Initialize lists to store variables for each ROI and iterate through each plant
    frame_avg = []
    yii_avg = []
    yii_std = []
    npq_avg = []
    npq_std = []
    plantarea = []
    ithroi = []
    inbounds = []
    if len(c) == 0:

        for i, rc in enumerate(roi_c):
            # each variable needs to be stored 2 x #roi
            frame_avg.append(np.nan)
            frame_avg.append(np.nan)
            yii_avg.append(np.nan)
            yii_avg.append(np.nan)
            yii_std.append(np.nan)
            yii_std.append(np.nan)
            npq_avg.append(np.nan)
            npq_avg.append(np.nan)
            npq_std.append(np.nan)
            npq_std.append(np.nan)
            inbounds.append(False)
            inbounds.append(False)
            plantarea.append(0)
            plantarea.append(0)
            # Store iteration Number even if there are no objects in image
            ithroi.append(int(i))
            ithroi.append(int(i))  # append twice so each image has a value.

    else:
        i = 1
        rc = roi_c[i]

        for i, rc in enumerate(roi_c):
            # Store iteration Number
            ithroi.append(int(i))
            ithroi.append(int(i))  # append twice so each image has a value.
            # extract ith hierarchy
            rh = roi_h[i]

            # Filter objects based on being in the defined ROI
            roi_obj, hierarchy_obj, submask, obj_area = pcv.roi_objects(
                img,
                roi_contour=rc,
                roi_hierarchy=rh,
                object_contour=c,
                obj_hierarchy=h,
                roi_type='partial')

            if obj_area == 0:
                print('!!! No plant detected in ROI ', str(i))

                frame_avg.append(np.nan)
                frame_avg.append(np.nan)
                yii_avg.append(np.nan)
                yii_avg.append(np.nan)
                yii_std.append(np.nan)
                yii_std.append(np.nan)
                npq_avg.append(np.nan)
                npq_avg.append(np.nan)
                npq_std.append(np.nan)
                npq_std.append(np.nan)
                inbounds.append(False)
                inbounds.append(False)
                plantarea.append(0)
                plantarea.append(0)

            else:

                # Combine multiple plant objects within an roi together
                plant_contour, plant_mask = pcv.object_composition(
                    img=img, contours=roi_obj, hierarchy=hierarchy_obj)

                #combine plant masks after roi filter
                if param_name == 'FvFm':
                    newmask = pcv.image_add(newmask, plant_mask)

                # Calc mean and std dev of fluoresence, YII, and NPQ and save to list
                frame_avg.append(masked_stats.mean(imgmin, plant_mask))
                frame_avg.append(masked_stats.mean(img, plant_mask))
                # need double because there are two images per loop
                yii_avg.append(masked_stats.mean(YII, plant_mask))
                yii_avg.append(masked_stats.mean(YII, plant_mask))
                yii_std.append(masked_stats.std(YII, plant_mask))
                yii_std.append(masked_stats.std(YII, plant_mask))
                npq_avg.append(masked_stats.mean(NPQ, plant_mask))
                npq_avg.append(masked_stats.mean(NPQ, plant_mask))
                npq_std.append(masked_stats.std(NPQ, plant_mask))
                npq_std.append(masked_stats.std(NPQ, plant_mask))
                plantarea.append(obj_area * pixelresolution**2)
                plantarea.append(obj_area * pixelresolution**2)

                # Check if plant is compeltely within the frame of the image
                inbounds.append(pcv.within_frame(plant_mask))
                inbounds.append(pcv.within_frame(plant_mask))

            # end try-except-else
        # end roi loop
    # end if there are objects from roi filter

    # save mask of all plants to file after roi filter
    if param_name == 'FvFm':
        pcv.print_image(newmask, os.path.join(maskdir, outfn + '-mask.png'))

    # Output a pseudocolor of NPQ and YII for each induction period for each image
    imgdir = os.path.join(outdir, 'pseudocolor_images', plantbarcode)
    os.makedirs(imgdir, exist_ok=True)
    npq_img = pcv.visualize.pseudocolor(NPQ,
                                        obj=None,
                                        mask=newmask,
                                        cmap='inferno',
                                        axes=False,
                                        min_value=0,
                                        max_value=2.5,
                                        background='black',
                                        obj_padding=0)
    npq_img = add_scalebar.add_scalebar(npq_img,
                                        pixelresolution=pixelresolution,
                                        barwidth=20,
                                        barlocation='lower left')
    # If you change the output size and resolution you will need to adjust the  timelapse video script
    npq_img.set_size_inches(6, 6, forward=False)
    npq_img.savefig(os.path.join(imgdir, outfn + '-NPQ.png'),
                    bbox_inches='tight',
                    dpi=150)
    npq_img.clf()

    yii_img = pcv.visualize.pseudocolor(YII,
                                    obj=None,
                                    mask=newmask,
                                    cmap=custom_colormaps.get_cmap(
                                        'imagingwin'),
                                    axes=False,
                                    min_value=0,
                                    max_value=1,
                                    background='black',
                                    obj_padding=0)
    yii_img = add_scalebar.add_scalebar(yii_img,
                                    pixelresolution=pixelresolution,
                                    barwidth=20,
                                    barlocation='lower left')
    yii_img.set_size_inches(6, 6, forward=False)
    yii_img.savefig(os.path.join(imgdir, outfn + '-YII.png'),
                bbox_inches='tight',
                dpi=150)
    yii_img.clf()


    # check YII values for uniqueness between all ROI. nonunique ROI suggests the plants grew into each other and can no longer be reliably separated in image processing.
    # a single value isn't always robust. I think because there are small independent objects that fall in one roi but not the other that change the object within the roi slightly.
    # also note, I originally designed this for trays of 2 pots. It will not detect if e.g. 2 out of 9 plants grow into each other
    rounded_avg = [round(n, 3) for n in yii_avg]
    rounded_std = [round(n, 3) for n in yii_std]
    if len(roi_c) > 1:
        isunique = not (rounded_avg.count(rounded_avg[0]) == len(yii_avg)
                        and rounded_std.count(rounded_std[0]) == len(yii_std))
    else:
        isunique = True

    # save all values to outgoing dataframe
    outdf['roi'] = ithroi
    outdf['frame_avg'] = frame_avg
    outdf['yii_avg'] = yii_avg
    outdf['npq_avg'] = npq_avg
    outdf['yii_std'] = yii_std
    outdf['npq_std'] = npq_std
    outdf['obj_in_frame'] = inbounds
    outdf['unique_roi'] = isunique

    return (outdf)


# end of function!


# %% save histogram data
def analyze_heterogeneity(img, mask, img_var, hist_bins = 128, hist_range=(0,1)):
    '''
    Function to analyze hetergeneity of plant.
    Inputs:
    img = grayscale image
    mask = binary image
    img_var = image name. used to store the data and label the histogram
    hist_bins = number of bins for histogram, default is 128.
    hist_range = range of the histogram given as a tuple, default is (0,1)
    Returns:
    hist = frequencies of histogram
    hist_bins = bin midpoints for frequencies
    hist_fig = histogram of data
    :param img: numpy.ndarray
    :param mask: numpy.ndarray
    :param img_var: str
    :param bins: int
    :param range: tuple
    :return hist: numpy.ndarray
    :return hist_bins: numpy.ndarray
    :return hist_fig: matplotlib figure
    '''
    import os
    import cv2
    import numpy as np
    import pandas as pd
    from plotnine import ggplot, geom_label, aes, geom_line
    from plantcv.plantcv import print_image
    from plantcv.plantcv import plot_image
    from plantcv.plantcv import fatal_error
    from plantcv.plantcv import params
    from plantcv.plantcv import outputs

    # Auto-increment the device counter
    params.device += 1
    # Check that img is grayscale and mask is binary
    if not all(len(np.shape(img)) == 2 for i in [img, mask]):
        fatal_error(
            "The image and mask must be grayscale images.")
    if not np.unique(mask)==2:
        fatal_error(
            "The mask should be binary.")

    hist, hist_bins = np.histogram(img[np.where(mask > 0)], hist_bins, range=hist_range)
    # hist_bins is a bins + 1 length list of bin endpoints, so we need to calculate bin midpoints so that
    # the we have a one-to-one list of x (FvFm) and y (frequency) values.
    # To do this we add half the bin width to each lower bin edge x-value
    midpoints = hist_bins[:-1] + 0.5 * np.diff(hist_bins)
    # Create Histogram Plot, if you change the bin number you might need to change binx so that it prints
    # an appropriate number of labels
    # Create a dataframe
    dataset = pd.DataFrame({'Plant Pixels': hist, 'Bins': midpoints})
    # Make the histogram figure using plotnine
    hist_fig = (
        ggplot(data=dataset, mapping=aes(x='Bins', y='Plant Pixels')) +
        geom_line(color='black', show_legend=True) +
        geom_label(label='Peak Bin Value: ' + str(max_bin),
                x=.15,
                y=205,
                size=8,
                color='black') +
        ggtitle(img_var))

    if params.debug == 'print':
        hist_fig.save(os.path.join(params.debug_outdir, str(params.device) + 'analyze_hetero_hist.png'))
    elif params.debug == 'plot':
        print(fvfm_hist_fig)

    outputs.add_observation(variable=img_var + '_frequencies',
                            trait=img_var + ' frequencies',
                            method='plantcv.plantcv.analyze_heterogeneity',
                            scale='none',
                            datatype=list,
                            value=hist.tolist(),
                            label=np.around(midpoints,
                                            decimals=len(
                                                str(hist_bins))).tolist())
    outputs.add_observation(variable=img_var + '_hist_peak',
                            trait='peak ' + img_var + ' value',
                            method='plantcv.plantcv.analyze_heterogeneity',
                            scale='none',
                            datatype=float,
                            value=float(max_bin),
                            label='none')
    outputs.add_observation(variable=img_var + '_median',
                            trait=img_var + ' median',
                            method='plantcv.plantcv.analyze_heterogeneity',
                            scale='none',
                            datatype=float,
                            value=float(np.around(np.median(hist), decimals=4)),
                            label='none')
    outputs.add_observation(variable=img_var + '_stdev',
                            trait=img_var + ' standard deviation',
                            method='plantcv.plantcv.analyze_hetergeneity',
                            scale='none',
                            datatype=float,
                            value=float(np.around(np.std(hist), decimals=4)),
                            label='none')

    return hist, hist_bins, hist_fig

# %% Setup Debug parameters
#by default params.debug should be 'None' when you are ready to process all your images
pcv.params.debug = None
# if you choose to print debug files to disk then remove the old ones first (if they exist)
if pcv.params.debug == 'print':
    import shutil
    shutil.rmtree(os.path.join(debugdir), ignore_errors=True)

# %% Testing dataframe
# # If you need to test new function or threshold values you can subset your dataframe to analyze some images
# df2 = df.query('((plantbarcode == "A6" or plantbarcode == "A3" or plantbarcode == "B3") and (parameter == "FvFm" or parameter == "t320_ALon" or parameter == "t300_ALon") and (jobdate == "2019-07-18" or jobdate == "2019-07-22"))')# | (plantbarcode == "B7" & jobdate == "2019-11-20")')
# # del df2
# fundf = df2.query('(plantbarcode == "A6" and parameter=="t40_ALon" and jobdate == "2019-07-18")')
# del fundf
# # # fundf
# # end testing

# %% Process the files
# check for subsetted dataframe
if 'df2' not in globals():
    df2 = df
else:
    print('df2 already exists!')

# removing this image because it causes python to crash with "Floating point exception (core dumped)"
df2 = df2.query('not (plantbarcode=="A6" and jobdate == "2019-07-18" and parameter=="t40_ALon")')

# # initialize db
# heterodb.insert({'plantbarcode': df2.plantbarcode.})
# heterodb.insert_multiple([{'plantbarcode': df2.plantbarcode.array}])
# heterodb.get('plantbarcode')
# heterodb.all()
# Each unique combination of treatment, plantbarcode, jobdate, parameter should result in exactly 2 rows in the dataframe that correspond to Fo/Fm or F'/Fm'
dfgrps = df2.groupby(['experiment', 'plantbarcode', 'jobdate', 'parameter'])
grplist = []
for grp, grpdf in dfgrps:
    # print(grp)#'%s ---' % (grp))
    grplist.append(image_avg(grpdf))
df_avg = pd.concat(grplist)

df_avg.to_csv('output/psII/df_avg.csv',na_rep='nan', float_format='%.4f', index=False)

# %% Add genotype information
gtypeinfo = pd.read_csv(os.path.join('data', 'genotype_map.csv'),
                        skipinitialspace=True)
df_avg2 = (pd.merge(df_avg, gtypeinfo, on=['plantbarcode', 'roi'], how='inner'))
# df_avg2.to_csv('wrongoutput2.csv')
# df_avg2 = (pd.merge(df_avg,
#                     gtypeinfo,
#                     left_on=['plantbarcode', 'roi'],
#                     right_on=['plantbarcode', 'roi'],
#                     how='inner'))
# df_avg2.to_csv('correctoutput.csv')


# %% Write the tabular results to file!
# df_avg2.jobdate = df_avg2.jobdate.dt.strftime('%Y-%m-%d')
(df_avg2.sort_values(['jobdate', 'plantbarcode', 'frameid']).drop(
    ['filename'], axis=1).to_csv(os.path.join(outdir,
                                              'output_psII_level0.csv'),
                                 na_rep='nan',
                                 float_format='%.4f',
                                 index=False))


# %%
