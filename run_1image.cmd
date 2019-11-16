
ipython -i -- scripts\visworkflow.py --image "data/raw_snapshots/vistest/C5-Lachowiec_VIS-2019-05-15 13_11_41-VIS0.tif" ^
--outdir output/vistest ^
--result output/vistest/vistest1.json ^
--regex "(.{2})-(.+)-(\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2})-(.+)" ^
--debug "plot" 
