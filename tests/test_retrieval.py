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
from retrieval_research.retrieval.graph import _number_values, _references, _section_aliases, _normalize_ocr_reference_text
from retrieval_research.retrieval.planner import plan_query
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

    def test_ocr_noise_expanded_patterns(self):
        cases = [
            ("Sectl0n 3 details", "section:3"),
            ("Secti0n 4 details", "section:4"),
            ("Chapt3r 5 covers", "section:5"),
            ("Chapt er 6 covers", "section:6"),
            ("Tab le 3 shows", "table:3"),
            ("Tab les 4 show", "table:4"),
            ("Tabke 5 shows", "table:5"),
            ("F1gure 6 illustrates", "figure:6"),
            ("F1gures 6-7 illustrate", "figure:6"),
            ("Flgure 7 illustrates", "figure:7"),
            ("ftgure 8 illustrates", "figure:8"),
            ("Figu re 9 shows", "figure:9"),
            ("Ligure 10 shows", "figure:10"),
            ("Equati0n 7 defines", "equation:7"),
            ("Eqnation 8 defines", "equation:8"),
            ("Arxi v:2401.12345", "arxiv:2401.12345"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                refs = _references(text)
                self.assertIn(expected, refs, f"{text!r} should yield {expected}")

    def test_ocr_normalize_reference_text_roundtrip(self):
        noisy = "Sectlon 2.1, Tab1e 3, F1g 4, Chapte r 5, ArXiv:2401.12345"
        normalized = _normalize_ocr_reference_text(noisy)
        refs = _references(normalized)
        self.assertIn("section:2.1", refs)
        self.assertIn("table:3", refs)
        self.assertIn("figure:4", refs)
        self.assertIn("section:5", refs)
        self.assertIn("arxiv:2401.12345", refs)

    def test_graph_stress_multiple_combined_noise(self):
        noisy = (
            "Sect1on 2.1 describes the method. "
            "Tab1e 1 and Tab1e 2 show results. "
            "F1gure 3 and Flg 4 illustrate architecture. "
            "Equat10n 5 defines loss. "
            "ArXiv:2401.12345 cites one paper."
        )
        refs = _references(noisy)
        for expected in (
            "section:2.1",
            "table:1", "table:2",
            "figure:3", "figure:4",
            "equation:5",
            "arxiv:2401.12345",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, refs, f"Missing {expected} in combined noise")

    def test_graph_stress_numeric_range_with_noise(self):
        noisy = "See F1gures 1-3 and Tab1es A1-A3."
        refs = _references(noisy)
        for i in range(1, 4):
            self.assertIn(f"figure:{i}", refs, f"figure:{i}")
            self.assertIn(f"table:a{i}", refs, f"table:a{i}")

    def test_graph_stress_section_hierarchy_with_noise(self):
        noisy = "Sect1on 3.2.1 describes the algorithm. Sectlon 3.2 covers setup."
        refs = _references(noisy)
        self.assertIn("section:3.2.1", refs)
        self.assertIn("section:3.2", refs)


class PlannerCalibrationTest(unittest.TestCase):
    def test_visual_queries_route_correctly(self):
        visual_queries = [
            ("show me the plot in this screenshot", "visual"),
            ("what does the diagram show", "visual"),
            ("describe the chart data", "visual"),
            ("find the screenshot of the interface", "visual"),
            ("interpret the flowchart", "visual"),
            ("what is shown in the illustration", "visual"),
            ("describe the ui layout", "visual"),
            ("show the infographic data", "visual"),
            ("is there a map of the region", "visual"),
            ("find the mockup of the dashboard", "visual"),
        ]
        for query, expected_type in visual_queries:
            with self.subTest(query=query):
                plan = plan_query(query)
                self.assertEqual(
                    plan.query_type, expected_type,
                    f"Query {query!r} expected {expected_type} got {plan.query_type}",
                )
                self.assertIn("visual", plan.routes)

    def test_table_queries_route_correctly(self):
        table_queries = [
            ("what is the total revenue", "table_or_form"),
            ("show the spreadsheet data", "table_or_form"),
            ("find the row for Q3", "table_or_form"),
            ("what are the column headers", "table_or_form"),
            ("list the budget categories", "table_or_form"),
            ("show the invoice line items", "table_or_form"),
            ("find the transaction records", "table_or_form"),
            ("what is the balance amount", "table_or_form"),
            ("check the payroll entry", "table_or_form"),
        ]
        for query, expected_type in table_queries:
            with self.subTest(query=query):
                plan = plan_query(query)
                self.assertEqual(
                    plan.query_type, expected_type,
                    f"Query {query!r} expected {expected_type} got {plan.query_type}",
                )
                self.assertIn("late", plan.routes)

    def test_graph_queries_route_correctly(self):
        graph_queries = [
            ("what topics are covered in section 3", "graph_nav"),
            ("list the references cited", "graph_nav"),
            ("find entities related to retrieval", "graph_nav"),
            ("show the document structure", "graph_nav"),
            ("what is the section hierarchy", "graph_nav"),
            ("find the bibliography entries", "graph_nav"),
            ("show the appendix contents", "graph_nav"),
            ("list the citations in chapter 2", "graph_nav"),
            ("navigate to subsection 4.1", "graph_nav"),
            ("what are the related terms in the glossary", "graph_nav"),
            ("show the table of contents", "table_or_form"),
        ]
        for query, expected_type in graph_queries:
            with self.subTest(query=query):
                plan = plan_query(query)
                self.assertEqual(
                    plan.query_type, expected_type,
                    f"Query {query!r} expected {expected_type} got {plan.query_type}",
                )
                if expected_type == "graph_nav":
                    self.assertIn("graph", plan.routes)

    def test_multihop_queries_route_correctly(self):
        multihop_queries = [
            ("compare the two approaches", "multi_hop"),
            ("what are the differences between methods", "multi_hop"),
            ("summarize the findings across documents", "multi_hop"),
            ("show the relationship between concepts", "multi_hop"),
            ("synthesize the results from all experiments", "multi_hop"),
            ("find common patterns across sections", "multi_hop"),
            ("compare results vs baseline", "multi_hop"),
            ("what are the similarities and differences", "multi_hop"),
            ("aggregate the metrics across documents", "multi_hop"),
            ("analyze trends across quarters", "multi_hop"),
        ]
        for query, expected_type in multihop_queries:
            with self.subTest(query=query):
                plan = plan_query(query)
                self.assertEqual(
                    plan.query_type, expected_type,
                    f"Query {query!r} expected {expected_type} got {plan.query_type}",
                )
                self.assertIn("hybrid", plan.routes)

    def test_identifier_queries_route_correctly(self):
        identifier_queries = [
            ("look up invoice INV-2024-001", "exact_lookup"),
            ("find document ID AB-12345", "exact_lookup"),
            ("search for patent US-9876543", "exact_lookup"),
            ("get the value for key ABC-123", "exact_lookup"),
            ("look up version 1.2.3", "exact_lookup"),
            ("find section DOI 10.1234/abc.def", "exact_lookup"),
            ("look up transaction TXN-9001", "exact_lookup"),
        ]
        for query, expected_type in identifier_queries:
            with self.subTest(query=query):
                plan = plan_query(query)
                self.assertEqual(
                    plan.query_type, expected_type,
                    f"Query {query!r} expected {expected_type} got {plan.query_type}",
                )
                self.assertEqual(plan.routes, ["bm25"])

    def test_semantic_queries_route_correctly(self):
        semantic_queries = [
            ("what is the main contribution of this work", "semantic"),
            ("explain how the method works", "semantic"),
            ("what are the key findings", "semantic"),
            ("describe the experimental setup", "semantic"),
            ("who developed this approach", "semantic"),
            ("why is this result important", "semantic"),
            ("what happens when the temperature increases", "semantic"),
            ("give me an overview of the approach", "semantic"),
            ("tell me about the dataset used", "semantic"),
        ]
        for query, expected_type in semantic_queries:
            with self.subTest(query=query):
                plan = plan_query(query)
                self.assertEqual(
                    plan.query_type, expected_type,
                    f"Query {query!r} expected {expected_type} got {plan.query_type}",
                )
                self.assertIn("dense", plan.routes)
                self.assertIn("bm25", plan.routes)

    def test_combined_intent_queries(self):
        table_multihop = plan_query("compare the totals across spreadsheets")
        self.assertEqual(table_multihop.query_type, "multi_hop")
        self.assertIn("bm25", table_multihop.routes)
        self.assertIn("hybrid", table_multihop.routes)
        self.assertIn("late", table_multihop.routes)

        graph_multihop = plan_query("compare sections across documents")
        self.assertEqual(graph_multihop.query_type, "multi_hop")
        self.assertIn("graph", graph_multihop.routes)
        self.assertIn("hybrid", graph_multihop.routes)

        visual_with_text = plan_query("describe the figure in section 3")
        self.assertEqual(visual_with_text.query_type, "visual")
        self.assertIn("visual", visual_with_text.routes)
        self.assertIn("hybrid", visual_with_text.routes)

        identifier_with_hint = plan_query("look up section ID 3.2.1")
        self.assertIn(identifier_with_hint.query_type, ("graph_nav", "exact_lookup"))
        self.assertTrue(identifier_with_hint.routes)

    def test_merge_strategy_validation(self):
        with self.assertRaises(ValueError):
            plan_query("test query", merge_strategy="invalid")


if __name__ == "__main__":
    unittest.main()
