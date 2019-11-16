
outdir=output/vis
mkdir -p $outdir

python \
$CONDA_PREFIX/bin/plantcv-workflow.py \
--dir data/raw_snapshots/vis \
--workflow scripts/visworkflow.py \
--type tif \
--json $outdir/vis.json \
--outdir $outdir \
--adaptor filename \
--delimiter "(.{2})-(.+)-(\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2})-(.+)" \
--timestampformat "%%Y-%%m-%%d %%H_%%M_%%S" \
--meta plantbarcode,measurementlabel,timestamp,camera \
--cpu 12 \
--writeimg \
--create

python $CONDA_PREFIX/bin/plantcv-utils.py json2csv -j $outdir/vis.json -c $outdir/vis.csv

