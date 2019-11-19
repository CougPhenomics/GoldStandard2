SET outdir=output/vistest
mkdir "%outdir%"

ipython %CONDA_PREFIX%/Scripts/plantcv-workflow.py -- ^
    --workflow scripts/visworkflow.py ^
    --dir data/testimages2/vis ^
    --type png ^
    --outdir %outdir% ^
    --json %outdir%/results.json ^
    --adaptor filename ^
    --delimiter - ^
    --meta plantbarcode,measurementlabel,timestamp,camera ^
    --cpu 4 ^
    --writeimg ^
    --other_args="--debugdir debug-images" ^
    --match plantbarcode:A6

ipython %CONDA_PREFIX%/Scripts/plantcv-utils.py -- json2csv --json %outdir%/results.json --csv %outdir%/results.csv
