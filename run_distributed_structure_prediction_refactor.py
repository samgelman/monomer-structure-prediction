import subprocess as sp
import time
import pandas as pd 
import numpy as np 
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future
import boto3 
import glob 
import os 

AF2_BIN_DIR="/pscratch/sd/v/vss2134/fresh_of1_env"
AF3_BIN_DIR="/global/homes/v/vss2134/miniforge3/envs/of3"
SP_OUTPUT = None
NGPUS=4
GPU_VRAM_GB=36

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
def sp_run(cmd_str, output = None):
    sp.run(cmd_str, shell = True, check = True, stdout = output, stderr = output)
    return

def download_file_from_s3_filepath(s3_path, outfile, s3_client):
    spl = s3_path.split("/")
    bucket = spl[2]
    key = "/".join(spl[3:])
    s3_client.download_file(bucket, key, outfile)
    return 

def sync_folder_to_s3(local_folder, bucket_name, s3_folder, s3_client):
    """
    Sync a local folder to an S3 bucket folder.

    Args:
        local_folder (str): Path to the local folder to upload.
        bucket_name (str): Name of the S3 bucket.
        s3_folder (str): Folder path in the S3 bucket where files will be uploaded.
    """
    
    # Traverse the local directory recursively
    for root, _, files in os.walk(local_folder):
        for file_name in files:
            # Create the full local path
            local_path = os.path.join(root, file_name)
            
            # Calculate the S3 path
            relative_path = os.path.relpath(local_path, local_folder)
            s3_path = os.path.join(s3_folder, relative_path).replace("\\", "/")  # Replace backslashes for S3 compatibility
            
            # Upload file to S3
            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_path}")
            s3_client.upload_file(local_path, bucket_name, s3_path)


def structure_prediction_pipeline(mgy_id, jackhmmer_s3_path, hhblits_s3_path, device_id, checkfile, session):
    sp_run(f"mkdir -p mgy_structure_pred_logs")
    log_handle = open(f"mgy_structure_pred_logs/{mgy_id}.log", "w+")
    outbucket_base = "s3://rcp-openfold-dataset-prd-361769554172/monomer_distill_clean_data_v3"
    outbucket_obs = f"{outbucket_base}/{mgy_id}"
    s3_client = session.client("s3")
    if Path(checkfile).exists():
        log_handle.write(f"Skipping {mgy_id} as checkfile exists\n")
        log_handle.close()
        return
    
    
    # download data and validate a3ms from s3
    start = time.time()
    sp_run(f"rm -rf /tmp/alignments/{mgy_id}")
    sp_run(f"mkdir -p /tmp/alignments/{mgy_id}")
    download_file_from_s3_filepath(jackhmmer_s3_path, f"/tmp/alignments/{mgy_id}/cfdb_uniref30_hits.a3m", s3_client)
    download_file_from_s3_filepath(hhblits_s3_path, f"/tmp/alignments/{mgy_id}/uniref100_hits.a3m", s3_client)
    ## validate MSAs
    ### make 2 subset a3ms so we dont need to parse the giant ones 
    sp_run(f"head -n 2 /tmp/alignments/{mgy_id}/uniref100_hits.a3m > /tmp/alignments/{mgy_id}/uniref100_hits_subset.a3m", log_handle)
    sp_run(f"head -n 2 /tmp/alignments/{mgy_id}/cfdb_uniref30_hits.a3m > /tmp/alignments/{mgy_id}/cfdb_uniref30_hits_subset.a3m", log_handle)
    sp_run(f"{AF3_BIN_DIR}/bin/python validate_msas.py /tmp/alignments/{mgy_id}", log_handle)
    sp_run(f"rm /tmp/alignments/{mgy_id}/uniref100_hits_subset.a3m /tmp/alignments/{mgy_id}/cfdb_uniref30_hits_subset.a3m")
    ## merge a3ms into a single file 
    sp_run(f"cat /tmp/alignments/{mgy_id}/uniref100_hits.a3m > /tmp/alignments/{mgy_id}/concat_cfdb_uniref100.a3m", log_handle)
    sp_run(f"tail -n+3 /tmp/alignments/{mgy_id}/cfdb_uniref30_hits.a3m >> /tmp/alignments/{mgy_id}/concat_cfdb_uniref100.a3m", log_handle)
    end = time.time()
    download_time = end - start
    ### filter a3ms
    start = time.time()
    sp_run(f"{AF2_BIN_DIR}/bin/mmseqs filtera3m /tmp/alignments/{mgy_id}/concat_cfdb_uniref100.a3m /tmp/alignments/{mgy_id}/concat_cfdb_uniref100_filtered.a3m --qid 0.0,0.2,0.4,0.6,0.8,1.0 --filter-min-enable 16000 --diff 5000 --qsc 0 --max-seq-id 0.95", log_handle)
    ## remove bc openfold inference script autoloads everyting in the alignment folder 
    sp_run(f" rm /tmp/alignments/{mgy_id}/uniref100_hits.a3m /tmp/alignments/{mgy_id}/cfdb_uniref30_hits.a3m /tmp/alignments/{mgy_id}/concat_cfdb_uniref100.a3m")
    end = time.time()
    filter_time = end - start
    
    ### generate hhsearch template:
    start = time.time()
    sp_run(f"{AF2_BIN_DIR}/bin/hhsearch -cpu 4 -i /tmp/alignments/{mgy_id}/concat_cfdb_uniref100_filtered.a3m -maxseq 1000000 -o /tmp/alignments/{mgy_id}/hhsearch_output.hhr -d /pscratch/sd/v/vss2134/pdb70/pdb70", log_handle)
    end = time.time()

    sp_run(f"mkdir -p /tmp/fa/{mgy_id}/")
    sp_run(f" head -n2 /tmp/alignments/{mgy_id}/concat_cfdb_uniref100_filtered.a3m > /tmp/fa/{mgy_id}/query.fa")
    hhsearch_template_time = end - start
    ### predict structures
    start = time.time()

    sp_run(f'''{AF2_BIN_DIR}/bin/python \
                run_pretrained_openfold.py \
                /tmp/fa/{mgy_id}/ \
                /global/cfs/cdirs/m4351/openfold_reference_datasets/pdb_data/mmcif_files/ \
                --model_device=cuda:{device_id} \
                --config_preset=model_1_ptm \
                --use_precomputed_alignments=/tmp/alignments \
                --jax_param_path=/global/homes/v/vss2134/openfold_orig/alphfold_params/params/params_model_1_ptm.npz,/global/homes/v/vss2134/openfold_orig/alphfold_params/params/params_model_2_ptm.npz \
                --output_dir /tmp/
                ''', log_handle)
    
    sp_run(f'''{AF2_BIN_DIR}/bin/python \
            run_pretrained_openfold.py \
            /tmp/fa/{mgy_id}/ \
            /global/cfs/cdirs/m4351/openfold_reference_datasets/pdb_data/mmcif_files/ \
            --model_device=cuda:{device_id} \
            --config_preset=model_3_ptm \
            --use_precomputed_alignments=/tmp/alignments \
            --jax_param_path=/global/homes/v/vss2134/openfold_orig/alphfold_params/params/params_model_3_ptm.npz,/global/homes/v/vss2134/openfold_orig/alphfold_params/params/params_model_4_ptm.npz,/global/homes/v/vss2134/openfold_orig/alphfold_params/params/params_model_5_ptm.npz \
            --output_dir /tmp/
            ''', log_handle)
    end = time.time()
    inference_time = end - start
    ## hmmsearch templates
    start = time.time()
    sp_run(f"bash generate_templates_af3.sh /tmp/alignments/{mgy_id}/", log_handle)
    end = time.time()
    hmmsearch_template_time = end - start
    ### select best model and upload to s3
    pred_score_files = glob.glob(f"/tmp/predictions/*/{mgy_id}*structure_metrics.csv")
    pred_score_files = [
        f"/tmp/predictions/params_model_1_ptm/{mgy_id}_model_1_ptm_structure_metrics.csv",
        f"/tmp/predictions/params_model_2_ptm/{mgy_id}_model_1_ptm_structure_metrics.csv",
        f"/tmp/predictions/params_model_3_ptm/{mgy_id}_model_3_ptm_structure_metrics.csv",
        f"/tmp/predictions/params_model_4_ptm/{mgy_id}_model_3_ptm_structure_metrics.csv",
        f"/tmp/predictions/params_model_5_ptm/{mgy_id}_model_3_ptm_structure_metrics.csv",
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
    
    sync_folder_to_s3(f"/tmp/alignments/{mgy_id}", "rcp-openfold-dataset-prd-361769554172", f"monomer_distill_clean_data_v3/{mgy_id}", s3_client)
    
    
    ### write out checkfile 
    time_dict = {"mgy_id": mgy_id, "download_time": download_time, "filter_time": filter_time, "template_time": hhsearch_template_time, "inference_time": inference_time, "hmmsearch_template_time": hmmsearch_template_time}
    time_df = pd.DataFrame([time_dict])
    time_df.to_csv(checkfile, index = False)
    print(f"Finished {mgy_id}")
    log_handle.close()
    return


def main():
    mem_limit_df = pd.DataFrame().assign(
        lbin = list(range(2,30,1)),
        mem_gb = np.linspace(6, 36, 28, dtype=int)
    )
    df = pd.read_csv("avail_long_seq_11.25.csv").assign(
        lbin = lambda x: (x.seqlen // 100).astype(int), 
        checkfile = lambda x: "mgy_structure_pred_timing/" + x.seqid + ".csv"
    ).merge(mem_limit_df).rename(columns = {"seqid": "mgy_id", "jackhmmer_msa_path": "jackhmmer_s3_path", "hhblits_msa_path": "hhblits_s3_path"}).sort_values("lbin")
    Path("mgy_structure_pred_timing/").mkdir(exist_ok=True)
    gpu_manager = GPUQueueManager(NGPUS, GPU_VRAM_GB)
    with ThreadPoolExecutor(max_workers=64) as executor: ## on a NERSC node at max we should have 24 possible concurrent jobs
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
                    #print(f"Starting {mgy_id} on GPU {gpu_id} allocating {mem_gb} GB")
                    sp_process = executor.submit(structure_prediction_pipeline, mgy_id, jackhmmer_s3_path, hhblits_s3_path, gpu_id, checkfile)
                    #sp_process = executor.submit(dfunc)
                    gpu_manager.add_job(Job(gpu_id, sp_process, mem_gb))
                    pending = False
                # else:
                #     print("waiting for jobs to finish")

                gpu_manager.check_job_finished()
                time.sleep(1)
    ## wait for all jobs to finish
    while not gpu_manager.is_complete():
        gpu_manager.check_job_finished()
        time.sleep(10)
# if __name__ == "__main__":
#     main()     
session = boto3.Session(profile_name="openfold")

structure_prediction_pipeline(
    "MGYP000005862325",
    "s3://rcp-openfold-dataset-prd-361769554172/jackhmmer_output_cfdb_distill_chunks/7331/MGYP000005862325.a3m",
    "s3://rcp-openfold-dataset-prd-361769554172/hhblits_output_cfdb_distill_chunks/0/MGYP000005862325.a3m",
    0,
    "test_MGYP000005862325.csv",
    session
)

# structure_prediction_pipeline(
#     "MGYP000040560362",
#     "s3://rcp-openfold-dataset-prd-361769554172/jackhmmer_output_cfdb_distill_chunks/1546/MGYP000040560362.a3m",
#     "s3://rcp-openfold-dataset-prd-361769554172/hhblits_output_cfdb_distill_chunks/0/MGYP000040560362.a3m",
#     0,
#     "test_MGYP000040560362.csv"
# )
    