"""IMPEdit: poisoning a multimodal RAG deployment by editing the retriever's
inter-modal projection.

    impedit.edit      training the projection edit (W' = R W0)
    impedit.payload   generating the poisoned images
    impedit.data      indexes and per-query target manifests
    impedit.eval      retrieval, generation, scoring
"""

__version__ = "1.0.0"
