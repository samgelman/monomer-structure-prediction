#%%
import boto3
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import ClientError
import os
from botocore.config import Config
from tqdm import tqdm
from pathlib import Path 
import sys 
import time 
# Initialize S3 client
session = boto3.Session(profile_name="openfold")
#retries={'max_attempts': 10, 'mode': 'adaptive'},
s3_client = session.client("s3", config=Config(max_pool_connections=72))
BASE_OUTDIR="/p/vast1/OpenFoldCollab/openfold-data/monomer-structure-prediction/monomer_msas_for_prediction"
def download_file(s3_path, local_file):
    """Download a single file from S3."""
    try:
        if Path(local_file).exists():
            return
        bucket = s3_path.split("/")[2]
        key = "/".join( s3_path.split("/")[3:])
        s3_client.download_file(bucket, key, local_file)
        return
    except ClientError as e:
        return f"Failed: {s3_path} with error {e}"


def download_row(row):
    try:
        mgy_id = row.mgy_id
        outdir = f"{BASE_OUTDIR}/{mgy_id}"
        Path(outdir).mkdir(parents = True, exist_ok = True)
        jackhmmer_s3_path = row.jackhmmer_s3_path
        download_file(jackhmmer_s3_path, 
        f"{outdir}/uniref100_hits.a3m"
        )
        hhblits_s3_path = row.hhblits_s3_path
        download_file(hhblits_s3_path,
        f"{outdir}/cfdb_uniref30_hits.a3m"
        )
    except:
        print("Likely rate limited. Skipping and sleeping 10")
        time.sleep(10)
        pass
    return 
    

def main(df, max_workers=70):
    # Read CSV to get S3 path
    rows = [i[1] for i in df.iterrows()]
    # Download files in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_s3_path = [executor.submit(download_row, row) for row in tqdm(rows, desc="Submitting")]
        for future in tqdm(as_completed(future_to_s3_path), total=len(rows), desc="Downloading"):
            results.append(future.result())
#%% 
if __name__ == "__main__":
    all_long_seq = pd.read_csv(sys.argv[1], names = ["mgy_id","hhblits_s3_path","jackhmmer_s3_path","seqlen"])
    main(all_long_seq)
    


# %%



#%%