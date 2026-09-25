"""Where the pipeline reads and writes. Every script imports its paths from here.

Override with environment variables rather than editing the scripts:

  CAREERNET_CACHE  corpus, embeddings and scratch files     default ~/.careernet-cache
  CAREERNET_SRC    folder holding the release CSVs           default $CAREERNET_CACHE/release
"""
import os
from pathlib import Path

CACHE = Path(os.environ.get("CAREERNET_CACHE", Path.home() / ".careernet-cache"))
SRC = Path(os.environ.get("CAREERNET_SRC", CACHE / "release"))
CORPUS = CACHE / "corpus"
EMB = CACHE / "emb"

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "docs"
