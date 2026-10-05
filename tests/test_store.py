from citetrace.paper.parser import PaperChunk
from citetrace.paper.store import JsonPaperStore


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
