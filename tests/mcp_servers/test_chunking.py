"""Structure-aware chunking tests. Token counting is injected as word count."""
from __future__ import annotations

from paperpilot.mcp_servers.colbert.chunking import (
    ChunkRecord,
    chunk_document,
    default_context_prefix,
)

WORD_COUNTER = lambda text: len(text.split())  # noqa: E731


def _para(n_words: int, filler: str = "word") -> str:
    return " ".join(f"{filler}{i}" for i in range(n_words))


def test_empty_text_yields_no_chunks():
    assert chunk_document("   \n  ", WORD_COUNTER) == []


def test_short_paragraphs_aggregate_into_one_chunk():
    text = f"{_para(4)}\n\n{_para(4)}"
    records = chunk_document(text, WORD_COUNTER, chunk_size=10, overlap=4)
    assert len(records) == 1
    assert records[0].text == f"{_para(4)}\n{_para(4)}"

def test_paragraph_boundary_not_cut_mid_paragraph():
    # Two 6-word paragraphs with chunk_size=10: neither paragraph fits
    # together with the other, so each becomes its own chunk, whole.
    text = f"{_para(6, 'alpha')}\n\n{_para(6, 'beta')}"
    records = chunk_document(text, WORD_COUNTER, chunk_size=10, overlap=4)
    assert len(records) == 2
    assert records[0].text == _para(6, "alpha")
    assert records[1].text == _para(6, "beta")


def test_oversized_paragraph_splits_into_sentence_windows():
    sentences = [
        "Sentence one talks about encoders.",
        "Sentence two covers pooling layers.",
        "Sentence three mentions results.",
        "Sentence four ends the section.",
    ]
    text = " ".join(sentences)  # 5-6 tokens per sentence
    records = chunk_document(text, WORD_COUNTER, chunk_size=12, overlap=8)
    assert len(records) >= 2
    # Sentence boundaries respected: every chunk starts at a sentence start.
    starts = {sentence.split()[0] for sentence in sentences}
    for record in records:
        assert record.text.split()[0] in starts


def test_small_trailing_piece_carried_as_overlap():
    # chunk_size=8: s1(4)+s2(4) fills a window; s2 (<= overlap 4) is carried
    # into the next window so consecutive windows share one sentence.
    s1 = "one two three four."
    s2 = "five six seven eight."
    s3 = "nine ten eleven twelve."
    text = f"{s1} {s2} {s3}"
    records = chunk_document(text, WORD_COUNTER, chunk_size=8, overlap=4)
    assert len(records) == 2
    assert records[0].text == f"{s1}\n{s2}"
    assert records[1].text == f"{s2}\n{s3}"


def test_heading_sets_section_hint_for_following_chunks():
    text = (
        "1 Introduction\n"
        f"{_para(6, 'intro')}\n\n"
        "2 Method\n"
        f"{_para(6, 'method')}"
    )
    records = chunk_document(text, WORD_COUNTER, chunk_size=10, overlap=4)
    assert len(records) >= 2
    assert records[0].section_hint == "1 Introduction"
    assert any(r.section_hint == "2 Method" for r in records)


def test_uppercase_line_is_heading():
    text = f"RELATED WORK\n{_para(6, 'cite')}"
    records = chunk_document(text, WORD_COUNTER, chunk_size=100)
    assert records[0].section_hint == "RELATED WORK"


def test_char_spans_are_ordered_and_within_bounds():
    text = f"{_para(6, 'a')}\n\n{_para(6, 'b')}\n\n{_para(6, 'c')}"
    records = chunk_document(text, WORD_COUNTER, chunk_size=10, overlap=0)
    spans = [tuple(r.char_span) for r in records]
    assert spans == sorted(spans)
    for start, end in spans:
        assert 0 <= start < end <= len(text)


def test_chunk_indexes_are_sequential():
    text = "\n\n".join(_para(6, f"p{i}") for i in range(4))
    records = chunk_document(text, WORD_COUNTER, chunk_size=10, overlap=0)
    assert [r.index for r in records] == list(range(len(records)))


def test_default_context_prefix_contains_source_section_and_part():
    record = ChunkRecord(
        text="body", section_hint="2 Method", char_start=0, char_end=4, index=2
    )
    prefix = default_context_prefix("arXiv:1234", record, total=7)
    assert prefix == "[source: paper arXiv:1234; section: 2 Method; part 3/7]"


def test_default_context_prefix_without_section():
    record = ChunkRecord(text="body", section_hint="", char_start=0, char_end=4, index=0)
    prefix = default_context_prefix("p1", record, total=1)
    assert prefix == "[source: paper p1; part 1/1]"
