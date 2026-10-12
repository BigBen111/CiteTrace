from citetrace.paper.parser import PaperChunk
from citetrace.paper.store import JsonPaperStore
import json
import pytest
from pathlib import Path


def test_store_search_returns_matching_chunk(tmp_path):
    store = JsonPaperStore(tmp_path)
    store.save(
        "paper",
        [
            PaperChunk("a", "paper", 1, "We collect approximately 1,000 robot trajectories."),
            PaperChunk("b", "paper", 2, "The visual encoder is initialized from pretrained weights."),
        ],
    )
    result = store.search("paper", "1000 robot trajectories", top_k=1)
    assert result
    assert result[0].page == 1


def test_exact_model_name_ranks_bibliography_entry_first(tmp_path):
    store = JsonPaperStore(tmp_path)
    store.save(
        "paper",
        [
            PaperChunk("noise", "paper", 1, "We compare several VLA models in homes."),
            PaperChunk(
                "reference",
                "paper",
                14,
                "[42] Kim et al. OpenVLA: An Open-Source Vision-Language-Action Model.",
            ),
        ],
    )

    result = store.search("paper", "文章里面有引用 OpenVLA 吗？", top_k=2)

    assert result
    assert result[0].source == "reference"
    assert result[0].page == 14


def test_search_cleans_pdf_control_characters(tmp_path):
    store = JsonPaperStore(tmp_path)
    store.save(
        "paper",
        [PaperChunk("a", "paper", 1, "OpenVLA\x00\x07 is listed as reference 42.")],
    )

    result = store.search("paper", "OpenVLA", top_k=1)

    assert result[0].text == "OpenVLA is listed as reference 42."


def test_legacy_index_and_metadata_remain_readable(tmp_path):
    chunk = PaperChunk("a", "paper", 1, "OpenVLA reference")
    (tmp_path / "paper.json").write_text(json.dumps([chunk.__dict__]))
    (tmp_path / "paper.meta.json").write_text(json.dumps({"total_pages": 7}))
    store = JsonPaperStore(tmp_path)
    assert store.load("paper") == [chunk]
    assert store._load_metadata("paper", [chunk])["total_pages"] == 7


def test_new_index_does_not_reuse_legacy_metadata(tmp_path):
    (tmp_path / "paper.meta.json").write_text(json.dumps({"total_pages": 99}))
    store = JsonPaperStore(tmp_path)
    chunks = [PaperChunk("a", "paper", 1, "OpenVLA reference")]
    store.save("paper", chunks)
    assert store._load_metadata("paper", chunks)["total_pages"] == 1


def test_failed_commit_preserves_index_metadata_and_cache(tmp_path, monkeypatch):
    store = JsonPaperStore(tmp_path)
    store.save("paper", [PaperChunk("a", "paper", 1, "original")], {"total_pages": 1})
    assert store.search("paper", "original")
    before = (tmp_path / "paper.json").read_bytes()
    def fail(*args):
        raise OSError("commit failure")
    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError):
        store.save("paper", [PaperChunk("b", "paper", 5, "new")], {"total_pages": 5})
    assert (tmp_path / "paper.json").read_bytes() == before
    assert store.search("paper", "original")
    assert list(tmp_path.iterdir()) == [tmp_path / "paper.json"]
