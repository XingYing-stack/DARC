import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# =========================
# paths
# =========================
model_path = "/data3/workhome/fanshengda/models/sentence-transformers/all-MiniLM-L6-v2"
data_path = "/share_data/data1/fanshengda/DEvo/data/solver_1212/solver_questioner_300_train_dedup.parquet"

# =========================
# load data
# =========================
data = pd.read_parquet(data_path).to_dict(orient="records")

questions = [
    sample["prompt"][1]["content"]
    for sample in data
]

# =========================
# load model
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(model_path, device=device)

# =========================
# encode all questions
# =========================
embeddings = model.encode(
    questions,
    batch_size=64,
    convert_to_tensor=True,
    normalize_embeddings=True,  # VERY important for cosine similarity
    show_progress_bar=True
)

# =========================
# similarity-based filtering
# =========================
threshold = 0.6

kept_indices = []
kept_embeddings = []

for idx, emb in tqdm(enumerate(embeddings), total=len(embeddings)):
    if len(kept_embeddings) == 0:
        kept_indices.append(idx)
        kept_embeddings.append(emb)
        continue

    sims = util.cos_sim(emb, torch.stack(kept_embeddings))[0]
    max_sim = sims.max().item()

    if max_sim < threshold:
        kept_indices.append(idx)
        kept_embeddings.append(emb)
    # else: filtered out

# =========================
# build filtered data
# =========================
filtered_data = [data[i] for i in kept_indices]

print(f"Original size: {len(data)}")
print(f"Filtered size: {len(filtered_data)}")
print(f"Removed: {len(data) - len(filtered_data)}")



# =========================
# save filtered data
# =========================
output_path = "/share_data/data1/fanshengda/DEvo/data/solver_1212/solver_questioner_300_train_dedup.parquet"

df_filtered = pd.DataFrame(filtered_data)
df_filtered.to_parquet(output_path, index=False)

print(f"Saved filtered data to: {output_path}")

