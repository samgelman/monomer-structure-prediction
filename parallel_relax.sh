#!/bin/bash 
set -e 
nworkers=32
idfile=avail_ids.txt
jobsPerWorker=$(( $(wc -l $idfile | awk '{print $1}') / $nworkers  + 1))

rm -rf relax_chunks/
mkdir -p relax_chunks/
split -l $jobsPerWorker $idfile relax_chunks/
i=0
for chunk in relax_chunks/*; do
    i=$((i+1))
    device=$((i % 4))
    echo "Running $chunk on cuda:$device"
    export CUDA_VISIBLE_DEVICES=$device
    python run_relax.py $chunk "cuda:$device" > $chunk.log 2>&1 & ## send all output to log file
    sleep 1 
done

wait 
