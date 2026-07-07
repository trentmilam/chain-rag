"""Load a Consilium Module directly from Qdrant -- no re-embedding at startup.

The ingest pipeline (ingest/run_ingest.py) embeds every chunk ONCE with the real
CUDA-backed embedder and writes it to Qdrant with a `chain` payload field. This
loader reads those precomputed vectors+text back out, one Module per chain, so
Consilium's proven Router/compose() code runs completely unmodified -- Qdrant is
the real embedding store, not a redundant side-index.
"""
from __future__ import annotations

from consilium.module import Chunk, Descriptor, Module


def load_module_from_qdrant(chain: str, client, collection: str, embedder, descriptor: Descriptor) -> Module:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    chunks: list[Chunk] = []
    offset = None
    flt = Filter(must=[FieldCondition(key="chain", match=MatchValue(value=chain))])
    while True:
        points, offset = client.scroll(
            collection_name=collection, scroll_filter=flt, limit=256,
            with_payload=True, with_vectors=True, offset=offset,
        )
        for p in points:
            payload = p.payload or {}
            doc_id = payload.get("doc_id")
            chunk_index = payload.get("chunk_index")
            if doc_id is None or chunk_index is None:
                raise ValueError(
                    f"malformed Qdrant point {p.id!r} in collection {collection!r} "
                    f"(chain={chain!r}): missing doc_id/chunk_index in payload -- "
                    "the ingest that wrote this point is broken; re-run ingest, don't serve from it"
                )
            if p.vector is None:
                raise ValueError(
                    f"Qdrant point {p.id!r} (doc={doc_id!r}, chain={chain!r}) has no vector -- "
                    "a partial/failed upsert; re-run ingest for this document"
                )
            chunks.append(Chunk(
                id=f"{doc_id}#{chunk_index}",
                doc=str(doc_id),
                text=str(payload.get("text", "")),
                vec=list(p.vector),
            ))
        if offset is None:
            break
    if not chunks:
        raise ValueError(
            f"no Qdrant points found for chain={chain!r} in collection={collection!r} "
            "-- run the ingest pipeline first"
        )
    return Module(name=descriptor.name, descriptor=descriptor, chunks=chunks, embedder=embedder)
