#!/usr/bin/env bash

source env_vars.txt

# untar the args (if tar file exists for them)
if [ -f "args.tar.gz" ]; then
    echo "Extracting args.tar.gz"
    tar -xf args.tar.gz
    rm args.tar.gz
fi

# untar the run output pre directories
if [ -f "run_output_pre.tar.gz" ]; then
    echo "Extracting run_output_pre.tar.gz"
    tar -xf run_output_pre.tar.gz
    rm run_output_pre.tar.gz
fi

# download the source data
if [ ! -f "code.tar.gz" ]; then
    echo "Downloading repository from GitHub..."
    python fetch_repo.py --github_url https://github.com/samgelman/monomer-structure-prediction --github_tag "$GITHUB_TAG" --github_token "$GITHUB_TOKEN"
else
    echo "Repository already downloaded, skipping..."
fi

echo "Submitting job via htcondor.sub..."
condor_submit htcondor.sub
