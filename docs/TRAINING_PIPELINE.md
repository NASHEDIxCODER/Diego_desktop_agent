# Diego NLP Training Pipeline

## Architecture

```
datasets/intents/*.json
         │
         ▼
   Load Examples ────► Validate Schema (30+ examples each)
         │
         ▼
   Compute dataset_hash (SHA256)
         │
         ▼
   Set embedding cache key (model + dataset_hash + text)
         │
         ▼
   Load SentenceTransformer (lazy singleton)
         │
         ▼
   Embed ALL texts in batches ──► batch_size=64
         │                         normalize_embeddings=True
         │                         show_progress_bar=True
         │                         heartbeat after every batch
         ▼
   Store in DuckDB (single transaction, bulk insert)
         │
         ▼
   Build Classifier Index
         │
         ▼
   Save Classifier ──► models/intent_classifier.pkl
   Save Metadata   ──► models/metadata.json
         │
         ▼
   Commit transaction
```

### Pipeline Stages

Every stage includes:
- **Timer**: Duration measurement (printed in TIMING logs)
- **Logging**: Structured JSON logs with stage name
- **Heartbeat**: Updates watchdog timestamp (stage name, not "idle")
- **Exception handling**: Rollback on failure, never partial state

### Stages

| Stage | Log Level | Description |
|-------|-----------|-------------|
| `loading_dataset` | INFO | Load JSON files from datasets/intents/ |
| `encoding_embeddings` | INFO | Generate embeddings with SentenceTransformer |
| `writing_database` | INFO | Bulk insert into DuckDB |
| `training_classifier` | INFO | Build FAISS/nearest-neighbor index |
| `saving_model` | INFO | Serialize classifier + metadata to disk |

## Watchdog

### Design

The watchdog is **heartbeat-based** — it never aborts while measurable progress is occurring.

- Thread runs in background, checks every 5 seconds
- Updates `_last_heartbeat` whenever a stage calls `heartbeat()`
- Abort only if NO heartbeat for the full timeout duration
- Default timeout: **600 seconds** (configurable via `TRAIN_TIMEOUT_SECONDS` in `.env`)
- No hardcoded 30-second timeout

### Heartbeat Locations

- Dataset loading complete
- After each embedding batch (64 texts)
- Database writes per intent
- Classifier fitting
- Model saving

### Abort Behavior

Sets `_ABORT_REASON` globally and stops the watchdog thread.
The training loop's `check_abort()` call detects the reason and raises `RuntimeError`.
The `finally` block rolls back the transaction cleanly.

## Embedding Cache

### Cache Key

`SHA256(embedding_model + dataset_hash + text)`

This ensures:
- Changing the model invalidates all cached embeddings
- Changing the dataset invalidates all cached embeddings
- Same model + same dataset + same text → cache hit

### Cache File

- Location: `data/embedding_cache.pkl`
- Format: Pickle dict of `{sha256_key: numpy_array}`
- Loaded lazily on first `embed()` call
- Saved to disk after batch encoding completes

### Cache Lifecycle

1. Trainer computes `dataset_hash` from all examples
2. Calls `set_dataset_hash(hash)` on the embeddings module
3. `_compute_cache_key()` includes hash in SHA256
4. New dataset hash → new keys → effectively invalidates old cache
5. Old entries remain on disk but are never matched

## Model Lifecycle

### Training (`python main.py --train`)

```
reset_guards()
set_heartbeat_callback()
watchdog.start()

store.begin_transaction()

  delete_derived_tables()
  load_datasets()          ─── 35+ intents, 30+ examples each
  compute_dataset_hash()
  set_dataset_hash()
  collect_all_texts()
  embed_batch(texts)       ─── batch_size=64, progress bar, heartbeats
  store_in_duckdb()        ─── bulk insert per intent
  classifier.build_index()
  classifier.save()
  save_metadata(..., training_time=...)

store.commit()

watchdog.stop()
```

### Startup (normal `python main.py`)

- **NEVER retrains automatically**
- Loads pre-saved classifier from `models/intent_classifier.pkl`
- Loads metadata from `models/metadata.json`
- Connects to DuckDB
- Loads plugins
- If model missing: displays `NLP model not found. Run: python main.py --train`

### Saved Artifacts

```
models/
├── intent_classifier.pkl   # Pickle: intents, embeddings, labels
└── metadata.json           # Training metadata
```

**metadata.json contains:**

| Field | Description |
|-------|-------------|
| `model_version` | Version string (e.g. "2.0.0") |
| `dataset_hash` | SHA256 of all intent examples |
| `embedding_model` | Model name (e.g. `all-MiniLM-L6-v2`) |
| `embedding_dim` | Embedding vector dimension (384) |
| `created_at` | ISO 8601 timestamp |
| `num_intents` | Number of intents trained |
| `num_examples` | Total example count |
| `similarity_threshold` | Min confidence for match |
| `training_time` | Wall-clock training duration (seconds) |

## Datasets

### Location

`datasets/intents/*.json`

### Format

```json
{
  "intent": "open_app",
  "description": "User wants to open an application",
  "examples": [
    "open calculator",
    "launch browser",
    "start firefox"
  ]
}
```

### Requirements

- Each file: 30+ realistic examples
- Valid JSON (use `json.load()` to verify)
- Examples must be non-empty strings
- Duplicates within a single file are allowed but warned about

### Intent Coverage

35 intents currently trained:
- greetings, exit, time/date queries
- 10 YouTube control intents
- 3 Telegram intents (send, read, reply)
- Brightness/volume controls
- open_app, take_note, read_note, system_info
- joke, news, help, who_am_i, weather_query

## Database

### Design

- Single DuckDB transaction for entire training run
- `delete_derived_tables()` clears `intent_embeddings`, `cached_embeddings`, `intent_examples` only
- Never touches `intents` parent table (IDs are stable)
- `add_intent()` is idempotent — reuses existing IDs
- Rollback only on actual exceptions (not watchdog timeout)

### Tables Used During Training

| Table | Purpose |
|-------|---------|
| `intents` | Intent definitions (stable, never dropped) |
| `intent_examples` | Training examples (cleared each training run) |
| `intent_embeddings` | Precomputed embeddings (cleared each run) |

## Performance Benchmarks

Measured on NVIDIA GPU (CUDA):

| Stage | Time |
|-------|------|
| Model loading (first time) | ~6s |
| Model loading (cached) | ~2s |
| Embedding generation (674 examples) | ~7s |
| Database writes (35 intents) | ~1s |
| Classifier build | <0.1s |
| Model serialization | <0.1s |
| Total training | ~8s |

### Without CUDA

If no GPU available, the SentenceTransformer model runs on CPU.
Expected times: 2-3x slower.

## CLI Commands

| Command | Description |
|---------|-------------|
| `python main.py --train` | Train NLP model from intent datasets |
| `python main.py --status` | Show model status and metadata |
| `python main.py --benchmark` | Run inference benchmark |
| `python main.py` | Normal startup (loads model, no training) |

## Troubleshooting

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `No module named 'sentence_transformers'` | Missing dependency | `pip install sentence-transformers` |
| `TFPreTrainedModel` import error | Transformers/TF conflict | Fixed by patching `_tf_available = False` in `embeddings.py` |
| DuckDB migration errors | Schema mismatch | Delete `data/Diego.duckdb` and retrain |
| `spaCy not available` | Optional dependency | Falls back to simple tokenizer (no action needed) |
| Model not found on startup | No trained model | Run `python main.py --train` |
| "training exceeded 30s" | Old timeout-based watchdog | Already fixed — heartbeat-based watchdog uses 600s timeout |
| DuckDB lock conflicts | Another process has DB open | Kill the other process or close other Python instances |

### Logs

All logs are structured JSON with fields:
- `timestamp` - ISO 8601
- `level` - ERROR/WARNING/INFO/DEBUG
- `logger` - Module name
- `message` - Human-readable message

### Performance Notes

- Embedding model loads on GPU (CUDA) when available
- Batch encoding (batch_size=64) for efficiency
- Bulk insert into DuckDB in single transaction
- Avoid duplicate SQL queries
- Avoid repeated model loading
- Heartbeat after every embedding batch prevents false watchdog aborts
- First classification call is slow (~6s) due to lazy model loading; subsequent calls are fast (~5ms)
</write_to_file>
<task_progress>
- [x] Fix watchdog `os._exit(1)` → proper abort flag + raise
- [x] Populate 4 dataset files with 30+ examples
- [x] Fix embed cache key (include dataset_hash)
- [x] Fix generate_datasets.py
- [x] Save training_time in metadata
- [x] Update TRAINING_PIPELINE.md
- [x] Run training and verify
- [x] Validate startup loads model without retraining
</task_progress>
</write_to_file>