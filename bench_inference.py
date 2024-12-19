#%%
import subprocess as sp
import time
import pandas as pd 
import numpy as np 
import boto3 
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import Future
from pathlib import Path

# GPU_VRAM=40000
# def assign_job_to_gpu(req_mem):
#     while True:
#         usage_per_device = [get_gpu_memory_usage(i) for i in range(4)]
#         for i in range(4):
#             if usage_per_device[i] + req_mem < GPU_VRAM:
#                 return i



def get_gpu_memory_usage(device_id=0):
    """Get current GPU memory in MB usage using nvidia-smi."""
    try:
        result = sp.run(
            ["nvidia-smi", f"--id={device_id}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stdout=sp.PIPE,
            stderr=sp.PIPE,
            text=True,
            check=True,
        )
        memory_usage = float(result.stdout.strip())
        return memory_usage
    except Exception as e:
        print(f"Error querying GPU memory: {e}")
        return -1.0 

def monitor_script(script_args, interval=1):
    """Run a script and monitor its GPU memory usage."""
    process = sp.Popen(script_args, stdout=sp.PIPE, stderr=sp.PIPE)
    peak_memory = 0

    try:
        while process.poll() is None:  # While the script is running
            memory_usage = get_gpu_memory_usage()
            current_peak = memory_usage
            if memory_usage:
                peak_memory = max(peak_memory, current_peak)
            time.sleep(interval)

        stdout, stderr = process.communicate()  # Get script output when it finishes
        print(f"Script output:\n{stdout.decode()}")
        if stderr:
            print(f"Script errors:\n{stderr.decode()}")
    except KeyboardInterrupt:
        process.terminate()
        print("Monitoring interrupted.")
    finally:
        process.wait()  # Ensure process is cleaned up
    print(f"Peak GPU memory usage: {peak_memory} MB")
    return peak_memory
session = boto3.Session(profile_name="openfold")
s3_client = session.client("s3")
        

baseline_memory = np.mean(get_gpu_memory_usage())
locdf = pd.read_csv("sbin_subset_11.25.csv")
bin_dir="/global/homes/v/vss2134/miniforge3/envs/of2"


def download_file_from_s3(full_obj_path, session, local_path):
    spl = full_obj_path.split("/")
    bucket_name = spl[2]
    obj_path = "/".join(spl[3:])
    s3 = session.client("s3")
    s3.download_file(bucket_name, obj_path, local_path)
    return 
#for i, row in locdf.iterrows():
def benchmark_inference(row, device):
    if Path(f"infernece_benchmark_wo_evo/{row['seqid']}.csv").exists():
        return device
    idx = row["seqid"]
    outdict = {"seqid": idx}
    ### download data
    #%%
    start = time.time()
    sp.run(f"mkdir -p /tmp/{idx}/", shell=True)
    download_file_from_s3(row["jackhmmer_msa_path"], session, f"/tmp/{idx}/uniref100_hits.a3m")
    download_file_from_s3(row["hhblits_msa_path"], session, f"/tmp/{idx}/cfdb_uniref30_hits.a3m")
    end = time.time()
    outdict["download_time"] = end - start
    #%%
    ### filter a3ms
    start = time.time()
    sp.run(f"{bin_dir}/bin/mmseqs filtera3m /tmp/{idx}/uniref100_hits.a3m /tmp/{idx}/uniref100_hits_filtered.a3m --qid 0.0,0.2,0.4,0.6,0.8,1.0 --filter-min-enable 1000 --diff 3000 --qsc 0 --max-seq-id 0.95", shell=True)
    sp.run(f"{bin_dir}/bin/mmseqs filtera3m /tmp/{idx}/cfdb_uniref30_hits.a3m /tmp/{idx}/cfdb_uniref30_hits_filtered.a3m --qid 0.0,0.2,0.4,0.6,0.8,1.0 --filter-min-enable 1000 --diff 3000 --qsc 0 --max-seq-id 0.95", shell=True)
    sp.run(f" rm -rf /tmp/{idx}/uniref100_hits.a3m", shell=True)
    sp.run(f" rm -rf /tmp/{idx}/cfdb_uniref30_hits.a3m", shell=True)
    end = time.time()
    outdict["filter_time"] = end - start
    #%%
    ### generate templates 
    start = time.time()
    sp.run(f"{bin_dir}/bin/hhsearch -i /tmp/{idx}/uniref100_hits_filtered.a3m -maxseq 1000000 -o /tmp/{idx}/hhsearch_output.hhr -d /pscratch/sd/v/vss2134/pdb70/pdb70", shell = True)


    #sp.run(f"bash repair_templates.sh /tmp/{idx}/")
    #%%
    sp.run(f"mkdir -p /tmp/fa/{idx}/", shell=True)
    sp.run(f" head -n2 /tmp/{idx}/cfdb_uniref30_hits_filtered.a3m > /tmp/fa/{idx}/query.fa", shell=True)
    end = time.time()
    outdict["template_time"] = end - start
    #%%
    ### run inference 
    start = time.time()
    script_args = [f"{bin_dir}/bin/python", 
                "run_pretrained_openfold.py", 
                f"/tmp/fa/{idx}/", 
                "/global/cfs/cdirs/m4351/openfold_reference_datasets/pdb_data/mmcif_files/", 
                f"--model_device=cuda:{device}",
                "--config_preset=model_2_ptm", 
                "--use_precomputed_alignments=/tmp/", 
                "--openfold_checkpoint_path=/global/homes/v/vss2134/openfold_orig/params/openfold_params/finetuning_ptm_2.pt"
                ]
    baseline_memory = np.mean(get_gpu_memory_usage())
    peak_memory = monitor_script(script_args)
    end = time.time()
    outdict["inference_time"] = end - start
    outdict["peak_memory"] = peak_memory - baseline_memory


    with open(f"infernece_benchmark_wo_evo/{idx}.csv", "w+") as f:
        outstr=""
        for key, val in outdict.items():
            outstr+=f"{key},{val}\n"
        f.write(f"{outstr}\n")
    return device

def main():
    avail_gpu_ids = [0, 1, 2, 3]
    queue = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, row in locdf.iterrows():
            while len(avail_gpu_ids) == 0:
                time.sleep(60)
            device  = avail_gpu_ids.pop()
            future = executor.submit(benchmark_inference, row, device)
            for future in queue:
                if future.done():
                    avail_gpu_ids.append(future.result())
                    queue.remove(future)
            


            

#%%
### add species annotation 
#sp.run(f"bash add_species_annotation_to_a3m.py /tmp/{idx}/uniref100_hits_filtered.a3m", shell=True)
