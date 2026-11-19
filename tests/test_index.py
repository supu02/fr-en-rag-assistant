import os
import stat

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_assistant import index


def _make_read_only(path):
    for root, dirs, files in os.walk(path):
        for name in files + dirs:
            p = os.path.join(root, name)
            os.chmod(p, os.stat(p).st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    os.chmod(path, os.stat(path).st_mode & ~stat.S_IWUSR)


def _make_writable(path):
    for root, dirs, _ in os.walk(path):
        os.chmod(root, os.stat(root).st_mode | stat.S_IWUSR)
        for name in dirs:
            os.chmod(os.path.join(root, name), os.stat(os.path.join(root, name)).st_mode | stat.S_IWUSR)


def test_read_only_index_is_opened_from_a_temp_copy(tmp_path, monkeypatch):
    """On AWS Lambda the image is read-only, but Qdrant local mode writes a lock file."""
    folder = tmp_path / "index" / "qdrant"
    client = QdrantClient(path=str(folder))
    client.create_collection("chunks", vectors_config=VectorParams(size=2, distance=Distance.COSINE))
    client.upsert("chunks", points=[PointStruct(id=0, vector=[1.0, 0.0], payload={"page_id": "F1"})])
    client.close()

    _make_read_only(folder)
    monkeypatch.setattr(index, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(index.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    try:
        client = index.get_client()
        hits = client.query_points("chunks", query=[1.0, 0.0], limit=1).points
        assert hits[0].payload["page_id"] == "F1"
        assert (tmp_path / "tmp" / "qdrant-index").is_dir()
        client.close()
    finally:
        _make_writable(folder.parent)
