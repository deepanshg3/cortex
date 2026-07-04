import os
import polars as pl
from tdc.multi_pred import DTI

def fetch_raw_binding_db():
    print("Contacting Therapeutics Data Commons servers...")
    
    # Fetch the BindingDB dataset
    data_loader = DTI(name='BindingDB_Kd')
    df_pandas = data_loader.get_data()
    
    print("Converting to Polars DataFrame...")
    df = pl.from_pandas(df_pandas)
    
    # FIX: Pointing explicitly to data/raw/ nested structure
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, "raw_binding_data.csv")
    
    print(f"Saving dataset to: {target_path}...")
    df.write_csv(target_path)
    
    print(f"\n[SUCCESS] Data safely isolated at: {target_path}")
    print(f"Total Lock-and-Key pairs downloaded: {df.shape[0]}")

if __name__ == "__main__":
    fetch_raw_binding_db()
    
