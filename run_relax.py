import pickle
from openfold.config import model_config
from openfold.utils.script_utils import relax_protein
import random 
import string
from pathlib import Path
import pandas as pd 
from tqdm import tqdm 
import boto3
import multiprocessing
import logging
logging.disable(logging.CRITICAL)
from multiprocessing import Pool
import os 
import sys 
# from deepspeed.utils import logger
# logger.setLevel(logging.CRITICAL)

def run_relax(mgy_id,device, session):
    s3_client = session.client("s3")
    bucket = "rcp-openfold-dataset-prd-361769554172"
    prefix = f"monomer_distill_clean_data_prod/{mgy_id}"

    ## check if the relaxed protein is already there
    try:
        s3_client.head_object(Bucket=bucket, Key=f"{prefix}/best_structure_relaxed.pdb")
        return
    except:
        pass
    rstring = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    tmp_dir = f"/tmp/{rstring}"
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    ## get the unrelaxed protein and the metric file 
    s3_client.download_file(bucket, f"{prefix}/best_structure_unrelaxed_protein.pkl", f"{tmp_dir}/unrelaxed_protein.pkl")
    s3_client.download_file(bucket, f"{prefix}/all_pred_metric.csv", f"{tmp_dir}/metric.csv")
    metric_df = pd.read_csv(f"{tmp_dir}/metric.csv")
    best_model = metric_df.sort_values("mean_plddt").tail(1).src_file.iloc[0]
    config_preset = "_".join(best_model.split("/")[-1].split("_")[1:4])

    config = model_config(
        config_preset, 
        long_sequence_inference=False,
        use_deepspeed_evoformer_attention=False
        )
    with open(f"{tmp_dir}/unrelaxed_protein.pkl", "rb") as f:
        unrelaxed_protein = pickle.load(f)
        output_name = "best_structure"
    relax_protein(config, device, unrelaxed_protein, tmp_dir, output_name,
                              False)
    s3_client.upload_file(f"{tmp_dir}/{output_name}_relaxed.pdb", bucket, f"{prefix}/{output_name}_relaxed.pdb")


def main():
    id_file = sys.argv[1]
    device = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]
    with open(id_file) as f:
        avail_ids = [line.strip() for line in f]
    session = boto3.Session(profile_name="openfold")
    for id in tqdm(avail_ids):
        run_relax(id, device, session)
if __name__ == "__main__":
    main()

# session = boto3.Session(profile_name="openfold")
# device = "cuda:1"
# run_relax("MGYP004412687409", device, session)
