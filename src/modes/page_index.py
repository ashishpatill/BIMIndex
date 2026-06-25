"""PageIndex tree-structured navigation for vectorless retrieval."""

from typing import Optional


class PageIndexNode:
    """A node in the PageIndex tree."""

    def __init__(
        self,
        node_id: str,
        label: str,
        node_type: str = "section",
        page_number: int = 0,
    ):
        self.id = node_id
        self.label = label
        self.type = node_type  # document, page, section, chunk
        self.page_number = page_number
        self.children: list["PageIndexNode"] = []
        self.parent: Optional["PageIndexNode"] = None
        self.content: str = ""
        self.relevance_score: float = 0.0


class PageIndex:
    """Tree-structured document index for navigation without dense vectors."""

    def __init__(self):
        self.root: Optional[PageIndexNode] = None
        self._nodes: dict[str, PageIndexNode] = {}

    def build_from_document(self, doc_id: str, pages: list[dict]):
        """Build PageIndex from parsed document pages with sections."""
        self.root = PageIndexNode(doc_id, f"Document {doc_id}", "document")
        self._nodes[doc_id] = self.root

        for page_data in pages:
            page_node = PageIndexNode(
                f"{doc_id}_p{page_data['number']}",
                f"Page {page_data['number']}",
                "page",
                page_data["number"],
            )
            page_node.content = page_data.get("text", "")
            page_node.parent = self.root
            self.root.children.append(page_node)
            self._nodes[page_node.id] = page_node

            for section in page_data.get("sections", []):
                section_node = PageIndexNode(
                    f"{page_node.id}_{section.get('heading', 'section')}",
                    section.get("heading", "Section"),
                    "section",
                    page_data["number"],
                )
                section_node.content = section.get("text", "")
                section_node.parent = page_node
                page_node.children.append(section_node)
                self._nodes[section_node.id] = section_node

    def get_node(self, node_id: str) -> Optional[PageIndexNode]:
        return self._nodes.get(node_id)

    def search_by_heading(self, heading: str) -> list[PageIndexNode]:
        """Find sections matching a heading."""
        results = []
        for node in self._nodes.values():
            if node.type == "section" and heading.lower() in node.label.lower():
                results.append(node)
        return results

    def search_by_page(self, page_number: int) -> list[PageIndexNode]:
        """Get all nodes on a specific page."""
        return [n for n in self._nodes.values() if n.page_number == page_number]

    def get_path(self, node_id: str) -> list[str]:
        """Get the tree path from root to node."""
        node = self._nodes.get(node_id)
        if not node:
            return []
        path = [node.label]
        while node.parent:
            node = node.parent
            path.append(node.label)
        return list(reversed(path))

    def traverse(self, node_id: Optional[str] = None) -> list[dict]:
        """DFS traversal returning flat list."""
        start = self._nodes.get(node_id) if node_id else self.root
        if not start:
            return []

        result = []
        stack = [start]
        while stack:
            node = stack.pop()
            result.append({
                "id": node.id,
                "label": node.label,
                "type": node.type,
                "page": node.page_number,
            })
            stack.extend(reversed(node.children))
        return result

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Simple text overlap search across all nodes."""
        scored = []
        query_lower = query.lower()
        for node in self._nodes.values():
            score = 0.0
            if query_lower in node.label.lower():
                score += 2.0
            if query_lower in node.content.lower():
                score += 1.0
            if score > 0:
                scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {"id": n.id, "label": n.label, "type": n.type, "page": n.page_number, "score": s}
            for n, s in scored[:top_k]
        ]
