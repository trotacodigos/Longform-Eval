"""
LaBSE-based context retrieval for document-level APE.

For each segment S_i in a document, retrieves:
  - top-k most similar segments (APEdoc-high)
  - bottom-k least similar segments (APEdoc-low)

Usage:
    python labse_retrieval.py --input data.jsonl --output output.jsonl --k 3
"""
from docape.utils import read_jsonl, write_jsonl

import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


@dataclass
class Segment:
    idx: int        # original position in document
    src: str        # source sentence
    tgt: str        # target (MT) sentence


@dataclass
class RetrievedContext:
    seg_idx: int                    # index of query segment
    src_seg: str
    tgt_seg: str
    high_ctx: List[Segment]         # APEdoc-high: top-k similar
    low_ctx: List[Segment]          # APEdoc-low:  bottom-k similar
    full_ctx: List[Segment]         # APEdoc-full: all except current


class LaBSEEncoder:
    """Encodes sentences using LaBSE and returns L2-normalized embeddings."""

    MODEL_NAME = "sentence-transformers/LaBSE"

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading LaBSE on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, sentences: List[str], batch_size: int = 64) -> np.ndarray:
        """Returns (N, D) float32 array of L2-normalized embeddings."""
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            output = self.model(**encoded)
            # Use [CLS] token embedding
            embeddings = output.last_hidden_state[:, 0, :]
            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().float().numpy())
        return np.vstack(all_embeddings)


def cosine_similarity(query: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """
    query:  (D,)
    corpus: (N, D)  — already L2-normalized
    returns (N,) similarity scores in [-1, 1]
    """
    return corpus @ query  # dot product == cosine sim for normalized vectors


def retrieve_context(
    segments: List[Segment],
    embeddings: np.ndarray,
    query_idx: int,
    k: int,
) -> Tuple[List[Segment], List[Segment], List[Segment]]:
    """
    Returns (high_ctx, low_ctx, full_ctx) for segment at query_idx.

    high_ctx: top-k most similar (excluding query), sorted by doc order
    low_ctx:  bottom-k least similar (excluding query), sorted by doc order
    full_ctx: all segments except query, in original doc order
    """
    n = len(segments)
    query_emb = embeddings[query_idx]

    # Indices of all segments except the query
    candidate_indices = [i for i in range(n) if i != query_idx]
    candidate_embs = embeddings[candidate_indices]

    # Cosine similarities
    sims = cosine_similarity(query_emb, candidate_embs)

    # Rank by similarity
    ranked = sorted(zip(candidate_indices, sims), key=lambda x: x[1], reverse=True)

    top_k_indices = sorted([idx for idx, _ in ranked[:k]])
    bottom_k_indices = sorted([idx for idx, _ in ranked[-k:]])
    full_indices = sorted(candidate_indices)

    high_ctx = [segments[i] for i in top_k_indices]
    low_ctx = [segments[i] for i in bottom_k_indices]
    full_ctx = [segments[i] for i in full_indices]

    return high_ctx, low_ctx, full_ctx


def process_document(
    doc_segments: List[dict],
    encoder: LaBSEEncoder,
    k: int,
) -> List[RetrievedContext]:
    """
    doc_segments: list of {"src": str, "tgt": str}
    Returns RetrievedContext for every segment in the document.
    """
    segments = [
        Segment(idx=i, src=s["src"], tgt=s["tgt"])
        for i, s in enumerate(doc_segments)
    ]

    # Encode all source sentences in one batch
    src_sentences = [seg.src for seg in segments]
    embeddings = encoder.encode(src_sentences)

    results = []
    for i, seg in enumerate(segments):
        high_ctx, low_ctx, full_ctx = retrieve_context(
            segments, embeddings, query_idx=i, k=k
        )
        results.append(
            RetrievedContext(
                seg_idx=i,
                src_seg=seg.src,
                tgt_seg=seg.tgt,
                high_ctx=high_ctx,
                low_ctx=low_ctx,
                full_ctx=full_ctx,
            )
        )
    return results


def format_context(ctx_segments: List[Segment]) -> Tuple[str, str]:
    """
    Converts a list of segments into (src_ctx, tgt_ctx) strings,
    preserving original document order (segments are already sorted by idx).
    """
    src_ctx = "\n".join(seg.src for seg in ctx_segments)
    tgt_ctx = "\n".join(seg.tgt for seg in ctx_segments)
    return src_ctx, tgt_ctx


def segment_to_dict(seg: Segment) -> dict:
    return asdict(seg)


def context_to_dict(ctx: RetrievedContext) -> dict:
    src_high, tgt_high = format_context(ctx.high_ctx)
    src_low, tgt_low = format_context(ctx.low_ctx)
    src_full, tgt_full = format_context(ctx.full_ctx)
    return {
        "seg_idx": ctx.seg_idx,
        "src_seg": ctx.src_seg,
        "tgt_seg": ctx.tgt_seg,
        "high": {"src_ctx": src_high, "tgt_ctx": tgt_high},
        "low":  {"src_ctx": src_low,  "tgt_ctx": tgt_low},
        "full": {"src_ctx": src_full, "tgt_ctx": tgt_full},
    }


def main():
    parser = argparse.ArgumentParser(
        description="LaBSE-based context retrieval for APE"
    )
    parser.add_argument(
        "--input", required=True,
        help=(
            "Input JSONL file. Each line: "
            '{"doc_id": str, "segments": [{"src": str, "tgt": str}, ...]}'
        ),
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSONL file with retrieved contexts per segment.",
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Number of segments to retrieve for high/low context (default: 3).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda', 'cpu', or None for auto-detect.",
    )
    args = parser.parse_args()

    encoder = LaBSEEncoder(device=args.device)
    documents = read_jsonl(args.input)

    output_records = []
    for doc in documents:
        doc_id = doc.get("doc_id", "unknown")
        doc_segments = doc["segments"]

        if len(doc_segments) <= args.k:
            print(
                f"[WARN] doc_id={doc_id} has only {len(doc_segments)} segments "
                f"(<= k={args.k}). Skipping."
            )
            continue

        print(f"Processing doc_id={doc_id} ({len(doc_segments)} segments)...")
        contexts = process_document(doc_segments, encoder, k=args.k)

        for ctx in contexts:
            record = {"doc_id": doc_id}
            record.update(context_to_dict(ctx))
            output_records.append(record)

    write_jsonl(args.output, output_records)
    print(f"Saved {len(output_records)} records to {args.output}")


if __name__ == "__main__":
    main()