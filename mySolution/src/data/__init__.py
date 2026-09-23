"""Dataset and batching. Produces the fixed batch object of PipelinePlan.md §2.5."""
from .chunks import (BATCH_KEYS, ChunkDataset, ChunkSpec, collate,  # noqa: F401
                     enumerate_chunks)
from .cond import build_cond  # noqa: F401
from .sampler import build_batches  # noqa: F401
