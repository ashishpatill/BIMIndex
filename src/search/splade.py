"""SPLADE sparse vector search for bag-of-words expansion retrieval."""

from typing import Optional


class SPLADESearcher:
    """SPLADE sparse retrieval using learned sparse representations."""

    def __init__(self, model_name: str = "naver/splade-cocondenser-selfdistil", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._index: dict[int, dict] = {}
        self._next_id = 0

    def _lazy_init(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(self.device)
        except ImportError:
            raise ImportError("transformers or torch not installed.")

    def encode(self, text: str) -> dict[str, float]:
        """Encode text into sparse bag-of-weights."""
        self._lazy_init()
        tokens = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            outputs = self._model(**tokens)
            logits = outputs.logits
            # Apply FLOPS regularization (ReLU then log)
            relu_logits = F.relu(logits)
            # Max over document length -> single vector per doc
            doc_vector = relu_logits.max(dim=1).values
            # Apply log transformation
            doc_vector = torch.log1p(doc_vector)
            # Remove entries below threshold
            doc_vector[doc_vector < 0.1] = 0

        # Convert to sparse dict
        nonzero_indices = doc_vector.squeeze().nonzero(as_tuple=True)[0]
        sparse_dict = {}
        for idx in nonzero_indices:
            token = self._tokenizer.decode([idx.item()])
            weight = doc_vector[0, idx].item()
            if weight > 0:
                sparse_dict[token] = weight

        return sparse_dict

    def index_document(self, doc_id: str, text: str):
        """Index a document by its SPLADE sparse vector."""
        sparse_vec = self.encode(text)
        self._index[self._next_id] = {
            "doc_id": doc_id,
            "sparse_vec": sparse_vec,
            "text": text,
        }
        self._next_id += 1

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search using sparse vector overlap (dot product)."""
        query_vec = self.encode(query)

        scores = []
        for idx, entry in self._index.items():
            overlap = 0.0
            for token, weight in query_vec.items():
                if token in entry["sparse_vec"]:
                    overlap += weight * entry["sparse_vec"][token]
            scores.append((overlap, entry))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {"doc_id": entry["doc_id"], "text": entry["text"], "score": score}
            for score, entry in scores[:top_k]
        ]
