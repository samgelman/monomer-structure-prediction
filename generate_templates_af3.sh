#!/bin/bash
set -e

alnDir=$1
cd $alnDir
if [ -e "hmm_output.sto" ]; then
    echo "file exists, skipping"
    exit 0
fi

envDir="/p/vast1/OpenFoldCollab/openfold-data/miniforge3/envs/amdof_relaxhip"
binDir="$envDir/bin"
## guarantee that that the query sequence is in the alignment
### generate the query sequence
mgyId=$(basename $alnDir)
head -n2  concat_cfdb_uniref100_filtered.a3m > query.fa
seqlen=$(tail -n1 query.fa | awk '{ print length($0) }')
### convert a3m to sto
## This seems a little excessive, but its the only way to actually get hmmsearch to work. 
### NOTE: need to double check that this 1) generates the correct unaligned seq and 2) that we get everything back when running jackhmmer 
$binDir/seqkit seq -u -w 0 concat_cfdb_uniref100_filtered.a3m | sed 's/-//g' > seqs_unaligned.fasta
$binDir/jackhmmer  -o /dev/null -A concat_cfdb_uniref100_filtered.sto --noali -N 1 -E 0.0001 --incE 0.0001 --F1 0.0005 --F2 0.00005 --F3 0.0000005 --cpu 4 query.fa seqs_unaligned.fasta
rm -f seqs_unaligned.fasta
## build inital hmm output
$binDir/hmmbuild --hand  --amino output.hmm concat_cfdb_uniref100_filtered.sto > /dev/null 2>&1
$binDir/hmmsearch --cpu 4 --noali --F1 0.1 --F2 0.1 --F3 0.1 -E 100 --incE 100 --domE 100 --incdomE 100 -A hmm_output_no_query.sto output.hmm /p/vast1/OpenFoldCollab/openfold-data/pdb_seqres.txt > /dev/null 2>&1

if [ ! -s "hmm_output_no_query.sto" ]; then
    echo "no hits found"
    rm -f output.hmm query.fa templates_with_query.fasta hmm_output_tmp.sto hmm_output_no_query.sto templates_no_query.tmp  hmm_output.tmp 
    touch hmm_output.sto
    touch EMPTY_TEMPLATE_OK
    exit 0
fi

rm -f concat_cfdb_uniref100_filtered.sto

cat query.fa > templates_with_query.fasta
$binDir/esl-reformat -u fasta hmm_output_no_query.sto >> templates_with_query.fasta
$binDir/hmmalign -o hmm_output_tmp.sto output.hmm templates_with_query.fasta
echo "# STOCKHOLM 1.0" > hmm_output.sto
echo "" >> hmm_output.sto
echo "#=GS $mgyId/1-$seqlen     DE [subseq from] mol:protein length:$seqlen  UNKNOWN" >> hmm_output.sto
tail -n+3  hmm_output_tmp.sto >> hmm_output.sto

rm -f output.hmm query.fa templates_with_query.fasta hmm_output_tmp.sto hmm_output_no_query.sto templates_no_query.tmp  hmm_output.tmp