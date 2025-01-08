#!/bin/bash 


max_chunks=10000
chunk_size=50
count=0

mkdir -p batched_jobs/
mkdir -p jackhmmer_logs/
touch started_jobs.txt
mkdir -p mgy_structure_pred_output
batchfile=batched_jobs/group_$count.sh
echo "#!/bin/bash" > $batchfile
chmod +x $batchfile
# Stage data with dbcast:

infile=$1 ## csv input file
mkdir "${infile}_chunks"
splits with split -d -a 6 -l 500  --additional-suffix=.csv $infile  "${infile}_chunks/chunk_"

find "${infile}_chunks/chunk_" -type f  | sort -V -r | head -n $max_chunks | while read file ;
do
    basefile=$(basename $file)
    echo "flux batch -o exit-timeout=none --output mgy_structure_pred_output/$basefile.out --nodes 1 --exclusive --time-limit 3.9h wrap_submission.sh $file" >> $batchfile
    echo "sleep 5" >> $batchfile

    ((count++))

    if (( count % chunk_size == 0 )); then
        echo "flux watch --all" >> $batchfile
        batchfile=batched_jobs/group_$count.sh
        echo "#!/bin/bash" > $batchfile
        chmod +x $batchfile
    fi
done

find batched_jobs/ -type f | while read p; do echo  "flux batch -o exit-timeout=none --nodes $chunk_size --exclusive --time-limit 4h $p"; echo "sleep 5" ; done > jobs.txt
