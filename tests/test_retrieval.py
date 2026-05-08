import tempfile
import unittest
from pathlib import Path

from retrieval_research.chunking import chunk_document
from retrieval_research.ingest import ingest_path
from retrieval_research.retrieval import (
    BM25Index,
    DenseIndex,
    LateInteractionIndex,
    GraphIndex,
    VisualPageIndex,
    build_indexes,
    search_document,
    search_corpus,
)
from retrieval_research.retrieval.graph import _number_values, _references, _section_aliases
from retrieval_research.schema import Chunk, Document, Page
from retrieval_research.storage import ArtifactStore


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = ArtifactStore(str(self.tmp / "data"))

        source = self.tmp / "doc.md"
        source.write_text(
            "# Test Document\n\n"
            "BM25 is useful for keyword search.\n"
            "Dense retrieval helps with semantic matches.\n"
            "Late interaction does fine-grained scoring.\n"
            "Graph retrieval finds entities and references.\n"
            "Section 2 describes the method.\n",
            encoding="utf-8",
        )
        self.document = ingest_path(str(source), store=self.store)
        self.chunks = chunk_document(self.document, max_words=15, overlap_words=2)
        self.store.save_chunks(self.document.id, self.chunks)
        build_indexes(self.store, self.document.id, mode="all")

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self.tmp))

    def test_bm25_search_returns_results(self):
        index = BM25Index.from_dict(self.store.load_index(self.document.id, "bm25"))
        hits = index.search("keyword BM25", top_k=3)
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0].retrieval_path, "bm25")

    def test_dense_search_returns_results(self):
        index = DenseIndex.from_dict(self.store.load_index(self.document.id, "dense"))
        hits = index.search("semantic match", top_k=3)
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0].retrieval_path, "dense")

    def test_late_search_returns_results(self):
        index = LateInteractionIndex.from_dict(self.store.load_index(self.document.id, "late"))
        hits = index.search("fine-grained scoring", top_k=3)
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0].retrieval_path, "late")

    def test_visual_search_returns_results(self):
        index = VisualPageIndex.from_dict(self.store.load_index(self.document.id, "visual"))
        hits = index.search("page visual", top_k=3)
        self.assertGreaterEqual(len(hits), 1)

    def test_graph_search_returns_results(self):
        index = GraphIndex.from_dict(self.store.load_index(self.document.id, "graph"))
        hits = index.search("Section 2 method", top_k=3)
        self.assertGreaterEqual(len(hits), 1)

    def test_search_document_returns_hits_for_all_modes(self):
        for mode in ("bm25", "dense", "late", "hybrid", "visual", "graph"):
            hits, steps = search_document(self.store, self.document.id, "retrieval", mode=mode, top_k=3)
            self.assertGreaterEqual(len(hits), 0, f"{mode} should return at least 0 hits")

    def test_search_document_planner_mode(self):
        hits, steps = search_document(
            self.store, self.document.id, "keyword BM25 retrieval", mode="planner", top_k=3
        )
        self.assertGreaterEqual(len(hits), 1)
        planner_merge = [s for s in steps if s["path"] == "planner_merge"]
        self.assertEqual(len(planner_merge), 1)

    def test_search_corpus_multi_document(self):
        source2 = self.tmp / "doc2.md"
        source2.write_text("# Second Doc\n\nAdditional keyword content about BM25.\n", encoding="utf-8")
        doc2 = ingest_path(str(source2), store=self.store)
        chunks2 = chunk_document(doc2, max_words=15, overlap_words=2)
        self.store.save_chunks(doc2.id, chunks2)
        build_indexes(self.store, doc2.id, mode="all")

        hits, steps = search_corpus(
            self.store, [self.document.id, doc2.id], "keyword BM25", mode="bm25", top_k=5
        )
        self.assertGreaterEqual(len(hits), 1)

    def test_corpus_search_cross_document_graph(self):
        source2 = self.tmp / "doc3.md"
        source2.write_text("# Bridge\n\nRelated keyword entity content.\n", encoding="utf-8")
        doc2 = ingest_path(str(source2), store=self.store)
        chunks2 = chunk_document(doc2, max_words=15, overlap_words=2)
        self.store.save_chunks(doc2.id, chunks2)
        build_indexes(self.store, doc2.id, mode="graph")

        hits, steps = search_corpus(
            self.store, [self.document.id, doc2.id],
            "keyword entity",
            mode="graph",
            top_k=5,
        )
        self.assertGreaterEqual(len(hits), 1)


class BM25UnitTest(unittest.TestCase):
    def test_empty_corpus_returns_empty(self):
        index = BM25Index([])
        hits = index.search("test", top_k=5)
        self.assertEqual(len(hits), 0)

    def test_empty_query_returns_empty(self):
        chunks = [Chunk(id="c1", document_id="d1", page_numbers=[1], text="some content", chunk_index=0)]
        index = BM25Index(chunks)
        hits = index.search("", top_k=5)
        self.assertEqual(len(hits), 0)

    def test_no_matching_terms_returns_empty(self):
        chunks = [Chunk(id="c1", document_id="d1", page_numbers=[1], text="alpha beta gamma", chunk_index=0)]
        index = BM25Index(chunks)
        hits = index.search("zzzzz", top_k=5)
        self.assertEqual(len(hits), 0)

    def test_respects_top_k(self):
        chunks = [
            Chunk(id=f"c{i}", document_id="d1", page_numbers=[1], text=f"keyword term {i}", chunk_index=i)
            for i in range(10)
        ]
        index = BM25Index(chunks)
        hits = index.search("keyword", top_k=3)
        self.assertLessEqual(len(hits), 3)


class DenseUnitTest(unittest.TestCase):
    def test_empty_corpus(self):
        index = DenseIndex([])
        hits = index.search("test", top_k=5)
        self.assertEqual(len(hits), 0)

    def test_identical_text_has_high_score(self):
        chunks = [Chunk(id="c1", document_id="d1", page_numbers=[1], text="same text here", chunk_index=0)]
        index = DenseIndex(chunks)
        hits = index.search("same text here", top_k=5)
        self.assertGreaterEqual(len(hits), 1)
        self.assertGreater(hits[0].score, 0.5)


class GraphUnitTest(unittest.TestCase):
    def test_builds_knowledge_graph(self):
        chunks = [
            Chunk(id="c1", document_id="d1", page_numbers=[1], text="Introduction to Acme Retrieval.", chunk_index=0, parent_section="Intro"),
            Chunk(id="c2", document_id="d1", page_numbers=[2], text="Acme Retrieval uses Table 1.", chunk_index=1, parent_section="Method"),
        ]
        index = GraphIndex(chunks)
        kg = index.knowledge_graph
        self.assertGreaterEqual(kg["stats"]["entity_count"], 1)
        self.assertGreaterEqual(kg["stats"]["reference_count"], 1)
        self.assertGreaterEqual(kg["stats"]["section_count"], 2)

    def test_corpus_graph_links_shared_entities(self):
        chunks_a = [
            Chunk(id="a1", document_id="doc_a", page_numbers=[1], text="SharedEntity is important.", chunk_index=0),
        ]
        chunks_b = [
            Chunk(id="b1", document_id="doc_b", page_numbers=[1], text="SharedEntity appears again.", chunk_index=0),
        ]
        index = GraphIndex(chunks_a + chunks_b)
        hits = index.search("SharedEntity", top_k=5)
        self.assertGreaterEqual(len(hits), 1)

    def test_number_values_expands_ranges(self):
        result = _number_values("1-3")
        self.assertIn("1", result)
        self.assertIn("2", result)
        self.assertIn("3", result)

        result = _number_values("A1-A3")
        self.assertIn("a1", result)
        self.assertIn("a2", result)
        self.assertIn("a3", result)

        result = _number_values("1, 2, 3")
        self.assertIn("1", result)
        self.assertIn("2", result)
        self.assertIn("3", result)

    def test_section_aliases_include_hierarchy(self):
        aliases = _section_aliases("3.2.1 Algorithm")
        self.assertIn("3.2.1 algorithm", aliases)
        self.assertIn("3.2.1", aliases)
        self.assertIn("3.2", aliases)
        self.assertIn("3", aliases)
        self.assertIn("algorithm", aliases)

        aliases = _section_aliases("2.1")
        self.assertIn("2.1", aliases)
        self.assertIn("2", aliases)

    def test_graph_references_expand_ranges(self):
        refs = _references("See Figures 1-3 for details.")
        for i in range(1, 4):
            self.assertIn(f"figure:{i}", refs, f"figure:{i} should be in refs")

        refs = _references("See Tables A1-A3 for data.")
        for i in range(1, 4):
            self.assertIn(f"table:a{i}", refs, f"table:a{i} should be in refs")

    def test_ocr_noise_new_patterns(self):
        refs = _references("Sect1on 3 details the method.")
        self.assertIn("section:3", refs)

        refs = _references("Equat10n 5 defines the loss.")
        self.assertIn("equation:5", refs)

        refs = _references("See ArXiv:2401.12345 for the preprint.")
        self.assertIn("arxiv:2401.12345", refs)


if __name__ == "__main__":
    unittest.main()
