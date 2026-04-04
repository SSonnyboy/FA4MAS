"""CHIEF 的可选 RAG 支持。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class OptionalRAGRetriever:
    """尽量加载检索器；缺失依赖或资源时自动退化为空结果。"""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.available = False
        self.records_gaia: List[Dict[str, Any]] = []
        self.records_assistant: List[Dict[str, Any]] = []

        base = base_dir or (Path(__file__).resolve().parents[3] / "CHIEF" / "rag")
        self._gaia_path = base / "kb" / "gaia_kb.json"
        self._assistant_path = base / "kb" / "assistantbench_kb.json"

        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            self._faiss = None
            self._model = None
            return

        gaia_index = base / "index" / "gaia.index"
        assistant_index = base / "index" / "assistantbench.index"
        if not (gaia_index.exists() and assistant_index.exists() and self._gaia_path.exists() and self._assistant_path.exists()):
            self._faiss = None
            self._model = None
            return

        try:
            self._faiss = faiss
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.index_gaia = faiss.read_index(str(gaia_index))
            self.index_assistant = faiss.read_index(str(assistant_index))
            with self._gaia_path.open("r", encoding="utf-8") as file:
                self.records_gaia = json.load(file)
            with self._assistant_path.open("r", encoding="utf-8") as file:
                self.records_assistant = json.load(file)
            self.available = True
        except Exception:
            self._faiss = None
            self._model = None
            self.available = False

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        if not self.available or self._faiss is None or self._model is None:
            return []

        vector = self._model.encode([query], convert_to_numpy=True)
        self._faiss.normalize_L2(vector)
        dist_gaia, idx_gaia = self.index_gaia.search(vector, top_k)
        dist_assistant, idx_assistant = self.index_assistant.search(vector, top_k)

        gaia_results = [
            {
                "source": "GAIA",
                "score": float(dist_gaia[0][index]),
                "question": self.records_gaia[idx_gaia[0][index]]["question"],
                "steps": self.records_gaia[idx_gaia[0][index]]["steps"],
            }
            for index in range(len(idx_gaia[0]))
        ]
        assistant_results = [
            {
                "source": "AssistantBench",
                "score": float(dist_assistant[0][index]),
                "text": self.records_assistant[idx_assistant[0][index]]["text"],
            }
            for index in range(len(idx_assistant[0]))
        ]
        return sorted(gaia_results + assistant_results, key=lambda item: item["score"], reverse=True)[:top_k]


def build_rag_text(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No retrieved examples available."
    blocks = []
    for index, item in enumerate(results, start=1):
        if item.get("source") == "GAIA":
            blocks.append(
                f"[RAG Example {index}]\nQuestion: {item.get('question')}\nAnnotated Steps:\n{item.get('steps')}"
            )
        else:
            blocks.append(f"[RAG Example {index}]\nText:\n{item.get('text')}")
    return "\n\n--- Retrieved Example ---\n\n".join(blocks)

