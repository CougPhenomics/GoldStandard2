
mkdir "output/vistest"

python.exe ^
%CONDA_PREFIX%\Scripts\plantcv-workflow.py ^
--dir data\raw_snapshots\vistest ^
--workflow scripts/visworkflow.py ^
--type tif ^
--json output/vistest/vistest.json ^
--outdir output/vistest ^
--adaptor filename ^
--delimiter "(.{2})-(.+)-(\d{4}-\d{2}-\d{2} \d{2}_\d{2}_\d{2})-(.+)" ^
--timestampformat "%%Y-%%m-%%d %%H_%%M_%%S" ^
--meta plantbarcode,measurementlabel,timestamp,camera ^
--writeimg ^
--create ^
--match plantbarcode:C5

python %CONDA_PREFIX%\Scripts\plantcv-utils.py json2csv -j output/vistest/vistest.json -c output/vistest/vistest.csv

