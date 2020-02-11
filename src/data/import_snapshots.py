# -*- coding: utf-8 -*-
import os
import glob
import re as re
from datetime import timedelta
import pandas as pd


def import_snapshots(snapshotdir, camera='vis'):
    '''
    Input:
    snapshotdir = directory of .tif files
    camera = the camera which captured the images. 'vis' or 'psii'

    Export .png into data/<camera> folder from LemnaBase using data-science-tools/LT-db-extractor.py
    for example: C6-GoldStandard_PSII-20190312T000911-PSII0-15.png
    '''

    # %% Get metadata from .tifs
    # snapshotdir = 'data/raw_snapshots/psII'

    fns = [fn for fn in glob.glob(pathname=os.path.join(snapshotdir, '*.png'))]
    fns

    flist = list()
    for fn in fns:
        f = re.split('[-]', os.path.splitext(os.path.basename(fn))[0])
        f.append(fn)
        flist.append(f)

    fdf = pd.DataFrame(flist,
                       columns=[
                           'plantbarcode', 'experiment', 'timestamp',
                           'cameralabel', 'frameid', 'filename'
                       ])

    # convert date and time columns to datetime format
    fdf['datetime'] = pd.to_datetime(fdf['timestamp'])
    fdf['jobdate'] = fdf.datetime.dt.floor('d')

    if camera.upper() == 'PSII':
        #create a jobdate to match dark and light measurements. dark experiments after 8PM correspond to the next day's light experiments
        fdf.loc[fdf.datetime.dt.hour >= 20,
                'jobdate'] = fdf.loc[fdf.datetime.dt.hour >= 20,
                                     'jobdate'] + timedelta(days=1)

        # convert image id from string to integer that can be sorted numerically
        fdf['frameid'] = fdf.frameid.astype('uint8')
        fdf = fdf.sort_values(['plantbarcode', 'datetime', 'frameid'])

    fdf = fdf.set_index(['plantbarcode', 'experiment', 'datetime',
                         'jobdate']).drop(columns=['timestamp'])

    # check for duplicate jobs of the same sample on the same day.  if jobs_removed.csv isnt blank then you shyould investigate!
    #dups = fdf.reset_index('datetime',drop=False).set_index(['frameid'],append=True).index.duplicated(keep='first')
    #dups_to_remove = fdf[dups].drop(columns=['frameid','filename']).reset_index().drop_duplicates()
    #dups_to_remove.to_csv('jobs_removed.csv',sep='\t')
    #

    return fdf
