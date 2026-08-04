import lancedb
import sys

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

db = lancedb.connect("./lancedb")
tbl = db.open_table("ashaar_baits")

print(f"Table size: {len(tbl):,}")
print("Building IVF_PQ index on vector column (this may take a few minutes)...")
# Creating an IVF-PQ index. 
# 256 partitions is a good default for ~3M rows, 
# 64 sub-vectors compresses the 2048 dims into 64 bytes.
tbl.create_index(metric="cosine", vector_column_name="vector", num_partitions=256, num_sub_vectors=64)
print("Vector index built successfully!")
