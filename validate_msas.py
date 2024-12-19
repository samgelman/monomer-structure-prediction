#%%
import sys 
from pathlib import Path
from openfold3.core.data.io.sequence.msa import parse_a3m
import sqlite3
def check_msa(file, ref_seq, parser, db, pdb_id):
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
        msa_array = parser(f.read())
    q_obs = "".join(msa_array.msa[0])
    if q_obs != ref_seq:
        raise ValueError(f"Query sequence does not match ground truth")

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
if __name__ == "__main__":
    msa_dir = sys.argv[1]
    mgy_id = msa_dir.split("/")[-1]
    #mgy_id="MGYP000040560362"
    db = "combined_short_long_monomer_distill.db"
    uniref100_file = f"{msa_dir}/uniref100_hits_subset.a3m"
    cfdb_file = f"{msa_dir}/cfdb_uniref30_hits_subset.a3m"
    ref_seq = get_seq_from_db(mgy_id, db)
    check_msa(uniref100_file, ref_seq, parse_a3m, "uniref100", mgy_id)
    check_msa(cfdb_file, ref_seq, parse_a3m, "cfdb", mgy_id)
    print("validate_msas.py: all MSAs OK")

