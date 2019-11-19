SET outdir=output/vistest
mkdir "%outdir%"

ipython scripts\visworkflow.py -- ^
--image "data/testimages/vis/A3-GoldStandard2_RGB-20190719T161032-VIS0.png" ^
--outdir %outdir% ^
--result %outdir%/vistest.json ^
--regex "(.{2})-(.+)-(\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})-(.+)" ^
--debug "print" ^
--debugdir "debug/vis"
