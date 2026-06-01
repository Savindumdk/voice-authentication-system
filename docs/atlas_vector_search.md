# MongoDB Atlas Vector Search (1:N speaker identification at scale)

The default `/identify` loads **every** enrolled embedding into memory and scores
them in one batched cosine op. That's fine for hundreds of users but doesn't scale
to tens of thousands. Atlas Vector Search runs an approximate nearest-neighbor
query server-side and returns only the top matches.

This is **off by default**. Enabling it requires creating a vector index in Atlas.

## 1. Create the vector index

In the Atlas UI: *Atlas Search → Create Search Index → JSON Editor → Vector Search*,
on the collection holding user voiceprints (`voice_auth.user_data`). Use:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 192,
      "similarity": "cosine"
    }
  ]
}
```

- `numDimensions`: **192** for ECAPA-TDNN. Change it if you switch
  `EMBEDDING_BACKEND` (e.g. CAM++ ≈ 512). Re-enroll users after switching.
- `similarity`: **must be `cosine`** — the app converts the Atlas score back to a
  raw cosine (`raw = 2*score - 1`) so it stays comparable to `VERIFICATION_THRESHOLD`.
- Name it to match `VECTOR_INDEX_NAME` (default `embedding_vector_index`).

## 2. Enable in the app

```env
USE_VECTOR_SEARCH=true
VECTOR_INDEX_NAME=embedding_vector_index
VECTOR_TOP_K=5
VECTOR_NUM_CANDIDATES=100   # ANN breadth: higher = more accurate, slower
```

## 3. Behavior

- `/identify` uses `$vectorSearch` for matching instead of the in-memory loop.
- If `HEAVY_PIPELINE_MODE=off`, it also **skips loading all embeddings**, so the
  request never pulls the full collection.
- If the query fails (missing index, transient error), `vector_search_users`
  returns `[]` and identification reports "not recognized" rather than crashing.
- Scores returned are raw cosine in `[-1, 1]`, compared against
  `VERIFICATION_THRESHOLD` exactly like the in-memory path.

## 4. Notes

- Atlas Vector Search requires an **Atlas** cluster (M10+ for production indexes);
  it is not available on a self-hosted/community MongoDB. For self-hosted, use a
  FAISS index loaded at startup instead.
- Keep the index dimension in sync with the active embedding backend.
