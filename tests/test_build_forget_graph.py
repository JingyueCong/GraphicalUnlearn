import importlib.util
import math
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_forget_graph.py"
SPEC = importlib.util.spec_from_file_location("build_forget_graph", SCRIPT_PATH)
GRAPH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GRAPH)


class ForgetGraphTest(unittest.TestCase):
    def test_related_documents_form_an_edge_and_weights_are_normalized(self):
        documents = [
            "Who wrote Book A Alice wrote Book A",
            "Which book was authored by Alice Alice authored Book A",
            "Where is Mountain Z Mountain Z is in Country Q",
        ]
        vectors = GRAPH.tfidf_vectors(documents)
        adjacency, edges = GRAPH.build_knn_graph(
            vectors, top_k=1, min_similarity=0.01
        )
        scores = GRAPH.weighted_pagerank(adjacency)
        weights = GRAPH.normalize_weights(scores, floor=0.25, ceiling=4.0)

        pairs = {(edge["source"], edge["target"]) for edge in edges}
        self.assertIn((0, 1), pairs)
        self.assertEqual(len(weights), 3)
        self.assertTrue(math.isclose(sum(weights) / len(weights), 1.0))
        self.assertTrue(all(weight > 0 for weight in weights))

    def test_singleton_graph(self):
        vectors = GRAPH.tfidf_vectors(["only one forget example"])
        adjacency, edges = GRAPH.build_knn_graph(vectors, top_k=8, min_similarity=0)
        scores = GRAPH.weighted_pagerank(adjacency)
        weights = GRAPH.normalize_weights(scores, floor=0.25, ceiling=4.0)

        self.assertEqual(edges, [])
        self.assertEqual(scores, [1.0])
        self.assertEqual(weights, [1.0])

    def test_community_detection_covers_every_node(self):
        edges = [
            {"source": 0, "target": 1, "weight": 1.0},
            {"source": 2, "target": 3, "weight": 1.0},
        ]
        communities, sizes = GRAPH.detect_communities(4, edges, seed=0)

        self.assertEqual(set(communities), {"0", "1", "2", "3"})
        self.assertEqual(communities["0"], communities["1"])
        self.assertEqual(communities["2"], communities["3"])
        self.assertEqual(sum(sizes.values()), 4)


if __name__ == "__main__":
    unittest.main()
