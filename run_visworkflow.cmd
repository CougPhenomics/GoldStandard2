python %CONDA_PREFIX%/Scripts/plantcv-workflow.py --workflow vis.py ^
    --dir data/raw_snapshots/vis ^
    --type png ^
    --outdir output/vis ^
    --json output/vis/results.json ^
    --adaptor filename ^
    --delimiter - ^
    --meta plantbarcode-measurementlabel-timestamp-camera ^
    -T 4 ^
    --writeimg ^
    --other_args="--debugdir debug-images" ^
    --create ^
    --match plantbarcode:A3

python %CONDA_PREFIX%/Scripts/plantcv-utils.py json2csv --json output/vis/results.json --csv output/vis/results.csv
