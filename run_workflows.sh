#! bash

outdir=output/vis
mkdir -p $outdir

python \
$CONDA_PREFIX/bin/plantcv-workflow.py \
--dir data/vis \
--workflow scripts/visworkflow.py \
--type png \
--json $outdir/vis.json \
--outdir $outdir \
--adaptor filename \
--delimiter "(.{2})-(.+)-(\d{8}T\d{6})-(.+)-(\d)" \
--timestampformat "%%Y%%m%%dT%%H%%M%%S" \
--meta plantbarcode,measurementlabel,timestamp,camera,id \
--cpu 8 \
--writeimg \
--create 
# --match plantbarcode:A6
# --dates 2019-07-17 \ #there were issues before the 17th

python $CONDA_PREFIX/bin/plantcv-utils.py json2csv -j $outdir/vis.json -c $outdir/vis.csv

