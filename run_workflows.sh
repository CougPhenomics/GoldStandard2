
outdir=output/vis
mkdir -p $outdir

python \
$CONDA_PREFIX/bin/plantcv-workflow.py \
--dir data/raw_snapshots/vis \
--workflow scripts/visworkflow.py \
--type png \
--json $outdir/vis.json \
--outdir $outdir \
--adaptor filename \
--delimiter "(.{2})-(.+)-(\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2})-(.+)" \
--timestampformat "%%Y%%m%%dT%%H%%M%%S" \
--meta plantbarcode,measurementlabel,timestamp,camera \
--cpu 12 \
--writeimg \
--create

python $CONDA_PREFIX/bin/plantcv-utils.py json2csv -j $outdir/vis.json -c $outdir/vis.csv

