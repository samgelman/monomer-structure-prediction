import subprocess as sp
import time
import pandas as pd 
import numpy as np 
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import Future

import glob 
import os 
import sqlite3
import string 
import sys
from datetime import datetime

AF2_BIN_DIR="/p/lustre5/swamy2/amdof_relaxhip"
BASE_MSA_DIR="/p/vast1/OpenFoldCollab/openfold-data/monomer-structure-prediction/monomer_msas_for_prediction"
OUTPUT_DIR="/p/vast1/OpenFoldCollab/openfold-data/monomer-structure-prediction/monomer_predicted_structures"

SP_OUTPUT = None
NGPUS=4
GPU_VRAM_GB=96

# code for parsing MSAs
## pulled from OF3 codebase but to retain compatibility w/ OF2 env 
def parse_fasta(fasta_string):
    """Parses FASTA file.

    This function needs to be wrapped in a with open call to read the file.

    Arguments:
        fasta_string:
            The string contents of a fasta file. The first sequence in the file
            should be the query sequence.

    Returns:
        tuple[Sequence[str], Sequence[str]]:
            A list of sequences and a list of metadata.
    """

    sequences = []
    metadata = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            metadata.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif line.startswith("#"):
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, metadata

def _msa_list_to_np(msa):
    """Converts a list of sequences to a numpy array.

    Args:
        msa (Sequence[str]):
            list of ALIGNED sequences of equal length.

    Returns:
        np.array:
            2D num.seq.-by-seq.len. numpy array
    """
    sequence_length = len(msa[0])
    msa_array = np.empty((len(msa), sequence_length), dtype="<U1")
    for i, sequence in enumerate(msa):
        msa_array[i] = list(sequence)
    return msa_array

def parse_a3m(msa_string, max_seq_count,):
    """Parses sequences and deletion matrix from a3m format alignment.

    This function needs to be wrapped in a with open call to read the file.

    Args:
        msa_string (str):
            The string contents of a a3m file. The first sequence in the file
            should be the query sequence.
        max_seq_count (int | None):
            The maximum number of sequences to parse from the file.

    Returns:
        Msa: A Msa object containing the sequences, deletion matrix and metadata.
    """

    sequences, metadata = parse_fasta(msa_string)
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)

    # Make the MSA matrix out of aligned (deletion-free) sequences.
    deletion_table = str.maketrans("", "", string.ascii_lowercase)
    msa = [s.translate(deletion_table) for s in sequences]

    # Embed in numpy array
    msa = _msa_list_to_np(msa)

    return msa

def check_msa(file, ref_seq, parser):
    """make sure MSA exists, and query sequence matches ground truth

    Args:
        file (str): abs path to MSA file
        ref_seq (str): ground truth amino acid sequence
        parser (Callable[str]): parsing function for MSA file. one of parse_a3m
                                or parse_stockholm
        db (str): which database MSA searched
        pdb_id (str): pdb_identifier for sequence. propagated from
                      input fasta

    Returns:
        str: logging info
    """
    if not Path(file).exists():
        raise FileNotFoundError(f"MSA file {file} not found")
    with open(file) as f:
        msa_array = parser(f.read(), max_seq_count = 1000000)
    q_obs = "".join(msa_array[0])
    if q_obs != ref_seq:
        print(q_obs)
        print(ref_seq)
        raise ValueError(f"Query sequence does not match ground truth")
    return 

def get_seq_from_db(mgy_id, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT sequence 
            FROM seqs
            WHERE seqid = ?
        """, (mgy_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None

def validate_msas(msa_dir, mgy_id):
    # msa_dir = sys.argv[1]
    # mgy_id = msa_dir.split("/")[-1]
    #mgy_id="MGYP000040560362"
    try:
        db = "combined_short_long_monomer_distill.db"
        uniref100_file = f"{msa_dir}/uniref100_hits_subset.a3m"
        cfdb_file = f"{msa_dir}/cfdb_uniref30_hits_subset.a3m"
        ref_seq = get_seq_from_db(mgy_id, db)
        check_msa(uniref100_file, ref_seq, parse_a3m)
        check_msa(cfdb_file, ref_seq, parse_a3m)
        return True
    except:
        return False


@dataclass
class Job:
    gpu_id: int
    sp_process: Future
    mem_gb : float

class GPUQueueManager:
    def __init__(self, ngpus, gpu_vram_gb):
        self.ngpus = ngpus
        self.gpu_vram_gb = gpu_vram_gb
        self.queue = []
        self.gpu_mem_used = [0 for i in range(ngpus)]
    def get_available_gpu(self, req_mem_gb):
        for i in range(self.ngpus):
            if self.gpu_mem_used[i] + req_mem_gb < self.gpu_vram_gb:
                return i
        return None
    def add_job(self, job):
        self.queue.append(job)
        self.gpu_mem_used[job.gpu_id] += job.mem_gb
        return
    def check_job_finished(self):
        for i in range(len(self.queue)):
            job = self.queue[i]
            if job.sp_process.done():
                #print(f"freeing up {job.mem_gb} GB on {job.gpu_id} GPU")
                self.gpu_mem_used[job.gpu_id] -= job.mem_gb
                self.queue.pop(i)
                return
        return
    def is_complete(self):
        return len(self.queue) == 0

## too lazy to use functools.partial
def sp_run(cmd_str, log_handle = None, failure_file=None, task=None):
    result = sp.run(cmd_str, shell = True, stdout = log_handle, stderr = log_handle)
    if result.returncode != 0 :
        if failure_file is not None and task is not None:
            with open(failure_file, "w+") as ofl:
                ofl.write(f"{task}\n")
        raise NotImplementedError
    return


def structure_prediction_pipeline(mgy_id, jackhmmer_s3_path, hhblits_s3_path, device_id, checkfile):

    sp_run(f"mkdir -p mgy_structure_pred_logs mgy_structure_pred_failures")
    log_handle = open(f"mgy_structure_pred_logs/{mgy_id}.log", "w+")
    failure_file = f"mgy_structure_pred_failures/{mgy_id}.log"
    if Path(checkfile).exists():
        log_handle.write(f"Skipping {mgy_id} as checkfile exists\n")
        log_handle.close()
        return
    
    
    # download data and validate a3ms from s3
    start = time.time()
    sp_run(f"rm -rf /tmp/alignments/{mgy_id}")
    sp_run(f"mkdir -p /tmp/alignments/{mgy_id}")
    ## validate MSAs
    ### make 2 subset a3ms so we dont need to parse the giant ones 
    sp_run(f"head -n 2 {BASE_MSA_DIR}/{mgy_id}/uniref100_hits.a3m > /tmp/alignments/{mgy_id}/uniref100_hits_subset.a3m", log_handle)
    sp_run(f"head -n 2 {BASE_MSA_DIR}/{mgy_id}/cfdb_uniref30_hits.a3m > /tmp/alignments/{mgy_id}/cfdb_uniref30_hits_subset.a3m", log_handle)
    is_valid = validate_msas(f"/tmp/alignments/{mgy_id}", mgy_id)
    if is_valid:
        log_handle.write("MSAs OK")
    else:
        with open(failure_file, "w+") as ofl:
            ofl.write("invalid_aln\n")
        return 

    sp_run(f"rm /tmp/alignments/{mgy_id}/uniref100_hits_subset.a3m /tmp/alignments/{mgy_id}/cfdb_uniref30_hits_subset.a3m")
    ## merge a3ms into a single file 
    sp_run(f"cat {BASE_MSA_DIR}/{mgy_id}/uniref100_hits.a3m > /tmp/alignments/{mgy_id}/concat_cfdb_uniref100.a3m", log_handle)
    sp_run(f"tail -n+3 {BASE_MSA_DIR}/{mgy_id}/cfdb_uniref30_hits.a3m >> /tmp/alignments/{mgy_id}/concat_cfdb_uniref100.a3m", log_handle)
    end = time.time()
    download_time = end - start
    ### filter a3ms
    start = time.time()
    sp_run(f"{AF2_BIN_DIR}/bin/mmseqs filtera3m /tmp/alignments/{mgy_id}/concat_cfdb_uniref100.a3m /tmp/alignments/{mgy_id}/concat_cfdb_uniref100_filtered.a3m --qid 0.0,0.2,0.4,0.6,0.8,1.0 --filter-min-enable 16000 --diff 3000 --qsc 0 --max-seq-id 0.95", log_handle, failure_file, "mmseqs_filter")
    ## remove bc openfold inference script autoloads everyting in the alignment folder 
    sp_run(f" rm  /tmp/alignments/{mgy_id}/concat_cfdb_uniref100.a3m")
    end = time.time()
    filter_time = end - start
    
    ### generate hhsearch template:
    start = time.time()
    sp_run(f"{AF2_BIN_DIR}/bin/hhsearch -cpu 4 -i /tmp/alignments/{mgy_id}/concat_cfdb_uniref100_filtered.a3m -maxseq 1000000 -o /tmp/alignments/{mgy_id}/hhsearch_output.hhr -d /p/vast1/OpenFoldCollab/openfold-data/pdb70/pdb70", log_handle, failure_file, "hhsearch")
    end = time.time()

    sp_run(f"mkdir -p /tmp/fa/{mgy_id}/")
    sp_run(f" head -n2 /tmp/alignments/{mgy_id}/concat_cfdb_uniref100_filtered.a3m > /tmp/fa/{mgy_id}/query.fa")
    hhsearch_template_time = end - start
    ### predict structures
    start = time.time()
    ## models seeds with templates
    sp_run(f'''{AF2_BIN_DIR}/bin/python \
                run_pretrained_openfold.py \
                /tmp/fa/{mgy_id}/ \
                /p/vast1/OpenFoldCollab/openfold-data/pdb_mmcif/mmcif_files/ \
                --model_device=cuda:{device_id} \
                --config_preset=model_1_ptm \
                --use_precomputed_alignments=/tmp/alignments \
                --jax_param_path=/p/vast1/OpenFoldCollab/openfold-amd/openfold/resources/params_model_1_ptm.npz \
                --max_template_date "2021-09-30" \
                --output_dir /tmp/
                ''', log_handle, failure_file, "structure_pred_with_templates")
    ## model seeds with out templates 
    sp_run(f'''{AF2_BIN_DIR}/bin/python \
            run_pretrained_openfold.py \
            /tmp/fa/{mgy_id}/ \
            /p/vast1/OpenFoldCollab/openfold-data/pdb_mmcif/mmcif_files/ \
            --model_device=cuda:{device_id} \
            --config_preset=model_3_ptm \
            --use_precomputed_alignments=/tmp/alignments \
            --jax_param_path=/p/vast1/OpenFoldCollab/openfold-amd/openfold/resources/params_model_3_ptm.npz \
            --max_template_date "2021-09-30" \
            --output_dir /tmp/
            ''', log_handle, failure_file, "structure_pred_no_templates")
    end = time.time()
    inference_time = end - start
    ## hmmsearch templates
    start = time.time()
    sp_run(f"bash generate_templates_af3.sh /tmp/alignments/{mgy_id}/ {AF2_BIN_DIR} ", log_handle, failure_file, "generate_templates_af3")
    if not Path(f"/tmp/alignments/{mgy_id}/hmm_output.sto").exists():
        with open(failure_file, "w+") as ofl:
            ofl.write("generate_templates_af3\n")
        return 


    end = time.time()
    hmmsearch_template_time = end - start
    ### select best model and upload to s3
    pred_score_files = glob.glob(f"/tmp/predictions/*/{mgy_id}*structure_metrics.csv")
    pred_score_files = [
        f"/tmp/predictions/{mgy_id}_model_1_ptm_structure_metrics.csv",
        f"/tmp/predictions/{mgy_id}_model_3_ptm_structure_metrics.csv",
    ]
    for f in pred_score_files:
        assert Path(f).exists(), f"{f} does not exist"
    #pred_score_files = glob.glob("/global/homes/v/vss2134/openfold3/testing/example_predictions/*structure_metrics.csv")
    all_pred_scores = pd.concat([
        pd.read_csv(f, names = ["mgy_id", "mean_plddt", "ptm"]).assign(
            src_file = f,
            model_id = f.split("/")[-1].split("_")[2]
        )
        for f in pred_score_files])
    all_pred_scores.to_csv(f"/tmp/alignments/{mgy_id}/all_pred_metric.csv", index = False)
    best_model = all_pred_scores.sort_values("mean_plddt").tail(1).src_file.iloc[0]
    sp_run(f"mkdir /tmp/alignments/{mgy_id}/extra_structures")
    sp_run(f"cp {best_model.replace('structure_metrics.csv', 'relaxed.pdb')} /tmp/alignments/{mgy_id}/best_structure_relaxed.pdb")
    

    sp_run(f"mkdir -p /tmp/alignments/{mgy_id}/extra_structures/")
    for file in pred_score_files:
        params = file.split("/")[3]
        sp_run(f"cp {file.replace('structure_metrics.csv', 'relaxed.pdb')} /tmp/alignments/{mgy_id}/extra_structures/{params}_relaxed.pdb")
    ## sync to outdir
    run_outdir = Path(f"{OUTPUT_DIR}/{mgy_id}")
    run_outdir.mkdir(exist_ok = True, parents = True)
    sp_run(f"rsync -av /tmp/alignments/{mgy_id}/ {OUTPUT_DIR}/{mgy_id}/ > /dev/null 2>&1")
    sp_run(f"rm -r /tmp/alignments/{mgy_id}/")
    
    ### write out checkfile 
    time_dict = {"mgy_id": mgy_id, "download_time": download_time, "filter_time": filter_time, "template_time": hhsearch_template_time, "inference_time": inference_time, "hmmsearch_template_time": hmmsearch_template_time}
    time_df = pd.DataFrame([time_dict])
    time_df.to_csv(checkfile, index = False)
    print(f"Finished {mgy_id}")
    log_handle.close()
    return


def main():
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"__SCRIPT_STARTED_AT_{formatted_now}__")
    mem_limit_df = pd.DataFrame().assign(
        lbin = [0,1,2,3,4,5,6,7,8,9],
        mem_gb = [10, 12,16, 20, 25, 32, 40, 50, 64, 80]
    )
    df = pd.read_csv(sys.argv[1], names = ["seqid","hhblits_msa_path","jackhmmer_msa_path","seqlen"]).assign(
        lbin = lambda x: (x.seqlen // 100).astype(int), 
        checkfile = lambda x: "mgy_structure_pred_timing/" + x.seqid + ".csv"
    ).merge(mem_limit_df, how = "left").assign(mem_gb = lambda x: x.mem_gb.fillna(80)).rename(columns = {"seqid": "mgy_id", "jackhmmer_msa_path": "jackhmmer_s3_path", "hhblits_msa_path": "hhblits_s3_path"})
    Path("mgy_structure_pred_timing/").mkdir(exist_ok=True)
    gpu_manager = GPUQueueManager(NGPUS, GPU_VRAM_GB)
    
    with ProcessPoolExecutor(max_workers=64) as executor: ## on a NERSC node at max we should have 24 possible concurrent jobs
        for _, row in df.iterrows():
            mgy_id = row["mgy_id"]
            jackhmmer_s3_path = row["jackhmmer_s3_path"]
            hhblits_s3_path = row["hhblits_s3_path"]
            checkfile = row["checkfile"]
            mem_gb = row["mem_gb"]
            pending = True 
            #structure_prediction_pipeline(mgy_id, jackhmmer_s3_path, hhblits_s3_path, 0, checkfile)
            while pending:
                gpu_id = gpu_manager.get_available_gpu(mem_gb)
                if gpu_id is not None:
                    print(f"Starting {mgy_id} on GPU {gpu_id} allocating {mem_gb} GB")
                    #structure_prediction_pipeline(mgy_id, jackhmmer_s3_path, hhblits_s3_path, gpu_id, checkfile, session)
                    sp_process = executor.submit(structure_prediction_pipeline, mgy_id, jackhmmer_s3_path, hhblits_s3_path, gpu_id, checkfile)
                    gpu_manager.add_job(Job(gpu_id, sp_process, mem_gb))
                    pending = False


                gpu_manager.check_job_finished()
                time.sleep(1)
    ## wait for all jobs to finish
    while not gpu_manager.is_complete():
        gpu_manager.check_job_finished()
        time.sleep(10)

if __name__ == "__main__":
    main()   
  
# session = boto3.Session(profile_name="openfold")
# df = pd.read_csv("has_nvida_pred.csv").assign(
#         lbin = lambda x: (x.seqlen // 100).astype(int), 
#         checkfile = lambda x: "mgy_structure_pred_timing/" + x.seqid + ".csv"
#     )
# row = df.iloc[0,:]

# structure_prediction_pipeline(
#     row.seqid,
#     row.jackhmmer_msa_path,
#     row.hhblits_msa_path,
#     0,
#     row.checkfile,
#     session 
# )
    
