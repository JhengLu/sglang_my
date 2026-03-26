"""
Tree parser for ChatML/HTML structured tokens.

Parses token sequences into a tree following the content's HTML/ChatML architecture.
Each leaf node of the tree becomes a "chunk" for centroid-based sparse attention.

Adapted from visualize_attention_flamegraph.py::parse_chatml_structure.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """A node in the ChatML/HTML token tree."""

    node_type: str  # 'root', 'chatml', 'html_tag', 'browser_element', 'text'
    label: str
    start_idx: int  # inclusive
    end_idx: int  # inclusive
    children: List[TreeNode] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def token_count(self) -> int:
        return self.end_idx - self.start_idx + 1

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = {
            "node_type": self.node_type,
            "label": self.label,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "token_count": self.token_count,
        }
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class FlatChunk:
    """A leaf node flattened into a simple (start, end) range."""

    start_idx: int  # inclusive
    end_idx: int  # inclusive
    chunk_id: int
    label: str

    @property
    def token_count(self) -> int:
        return self.end_idx - self.start_idx + 1

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "chunk_id": self.chunk_id,
            "label": self.label,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "token_count": self.token_count,
        }


def parse_chatml_tree(token_texts: List[str]) -> TreeNode:
    """
    Parse ChatML/HTML structured tokens into a tree.

    Recognizes:
    - ChatML markers: <|im_start|>role ... <|im_end|>
    - HTML tags: <tag>...</tag> and self-closing <tag />
    - Browser state with tab-based indentation
    - Text content nodes

    Returns the root TreeNode of the parsed tree.
    """
    n = len(token_texts)
    root = TreeNode("root", "all", 0, n - 1 if n > 0 else 0)

    if n == 0:
        return root

    current_chatml_node = None
    current_parent = root
    stack = [root]
    browser_state_mode = False
    browser_depth_stack = [root]

    i = 0
    while i < n:
        token = token_texts[i]

        # --- Vision start/end markers (Qwen VL models) ---
        if "<|vision_start|>" in token:
            vision_start = i
            i += 1
            while i < n and "<|vision_end|>" not in token_texts[i]:
                i += 1
            vision_end = i if i < n else n - 1
            node = TreeNode("vision", "image_tokens", vision_start, vision_end)
            current_parent.children.append(node)
            i = vision_end + 1
            continue

        # --- ChatML start markers ---
        if "<|im_start|>" in token:
            role_parts = []
            role_idx = i + 1
            while role_idx < n and "<|im_start|>" in token_texts[role_idx]:
                role_idx += 1
            while role_idx < n and token_texts[role_idx] not in ["\n", "<", "|"]:
                role_parts.append(token_texts[role_idx])
                role_idx += 1
                if len(role_parts) > 10:
                    break
            role = "".join(role_parts).strip() or "unknown"
            node = TreeNode("chatml", role, i, role_idx)
            root.children.append(node)
            current_chatml_node = node
            current_parent = node
            stack = [root, node]
            browser_state_mode = False
            i = role_idx
            continue

        # --- ChatML end markers ---
        if "<|im_end|>" in token:
            if current_chatml_node:
                if i + 1 < n and token_texts[i + 1] == "\n":
                    current_chatml_node.end_idx = i + 1
                    i += 2
                else:
                    current_chatml_node.end_idx = i
                    i += 1
                current_chatml_node = None
                current_parent = root
                stack = [root]
            else:
                i += 1
            browser_state_mode = False
            continue

        # --- Tab-based indentation detection for browser state ---
        tab_count = 0
        if i > 0 and token_texts[i - 1] == "\t":
            check_idx = i - 1
            while check_idx >= 0 and token_texts[check_idx] == "\t":
                tab_count += 1
                check_idx -= 1
            if check_idx >= 0 and "\n" in token_texts[check_idx]:
                browser_state_mode = True

        # --- Browser state handling ---
        if browser_state_mode:
            # Closing tags
            if token == "</" or token.startswith("</"):
                j = i
                tag_end = min(i + 5, n)
                while j < tag_end and ">" not in token_texts[j]:
                    j += 1
                if j < n and ">" in token_texts[j]:
                    tag_str = "".join(token_texts[i : j + 1])
                    closing_match = re.match(r".*?</(\w+)", tag_str)
                    if closing_match and len(stack) > 1:
                        closing_name = closing_match.group(1)
                        if closing_name == "browser_state":
                            browser_state_mode = False
                        for idx in range(len(stack) - 1, 0, -1):
                            if (
                                stack[idx].node_type == "html_tag"
                                and closing_name in stack[idx].label
                            ):
                                stack[idx].end_idx = j
                                while len(stack) > idx:
                                    stack.pop()
                                current_parent = stack[-1]
                                break
                    i = j + 1
                    continue

            # Skip markers like |SCROLL|
            if token.startswith("|"):
                j = i
                marker_end = min(i + 10, n)
                while j < marker_end:
                    if j > i and "|" in token_texts[j]:
                        break
                    j += 1
                if j < n:
                    marker_str = "".join(token_texts[i : j + 1])
                    if marker_str.startswith("|") and "|" in marker_str[1:]:
                        i = j + 1
                        continue

            # Browser elements: [1501]<div /> or *[2352]<a />
            if token in ("[", "*") or token.startswith("[") or token.startswith("*"):
                elem_tab_count = 0
                check_idx = i - 1
                while check_idx >= 0:
                    if "\t" in token_texts[check_idx]:
                        elem_tab_count += token_texts[check_idx].count("\t")
                        check_idx -= 1
                    elif "\n" in token_texts[check_idx]:
                        break
                    else:
                        break

                test_str = "".join(token_texts[i : min(i + 10, n)])
                if re.match(r"^\*?\[\d+\]<", test_str):
                    j = i
                    tag_end = min(i + 25, n)
                    found_end = False
                    while j < tag_end:
                        if "/>" in token_texts[j]:
                            found_end = True
                            break
                        if token_texts[j] == ">" or (
                            token_texts[j].endswith(">")
                            and not token_texts[j].endswith("/>")
                        ):
                            found_end = True
                            break
                        j += 1

                    if found_end and j < n:
                        elem_str = "".join(token_texts[i : j + 1])
                        k = j + 1
                        while k < n and token_texts[k] not in ["\n", "\t"]:
                            next_test = "".join(token_texts[k : min(k + 10, n)])
                            if next_test.startswith("<") or re.match(
                                r"^\*?\[\d+\]<", next_test
                            ):
                                break
                            k += 1

                        display = (
                            elem_str if len(elem_str) <= 50 else elem_str[:47] + "..."
                        )
                        node = TreeNode("browser_element", display, i, k - 1)

                        while len(browser_depth_stack) <= elem_tab_count:
                            browser_depth_stack.append(browser_depth_stack[-1])
                        browser_depth_stack = browser_depth_stack[
                            : elem_tab_count + 1
                        ]

                        parent_node = browser_depth_stack[elem_tab_count]
                        parent_node.children.append(node)

                        if len(browser_depth_stack) == elem_tab_count + 1:
                            browser_depth_stack.append(node)
                        else:
                            browser_depth_stack[elem_tab_count + 1] = node

                        i = k
                        continue

            # <html /> tags in browser state
            if token.startswith("<html"):
                j = i
                while j < n and ">" not in token_texts[j]:
                    j += 1
                if j < n:
                    tag_str = "".join(token_texts[i : j + 1])
                    k = j + 1
                    while k < n and token_texts[k] not in ["\n", "\t"]:
                        if re.match(
                            r"^\*?\[\d+\]<", token_texts[k]
                        ) or token_texts[k].startswith("<"):
                            break
                        k += 1

                    display = (
                        tag_str if len(tag_str) <= 50 else tag_str[:47] + "..."
                    )
                    node = TreeNode("html_tag", display, i, k - 1)

                    while len(browser_depth_stack) <= tab_count:
                        browser_depth_stack.append(browser_depth_stack[-1])
                    browser_depth_stack = browser_depth_stack[: tab_count + 1]
                    parent_node = browser_depth_stack[tab_count]
                    parent_node.children.append(node)

                    if len(browser_depth_stack) == tab_count + 1:
                        browser_depth_stack.append(node)
                    else:
                        browser_depth_stack[tab_count + 1] = node

                    i = k
                    continue

        # --- Regular HTML/XML tag parsing (non-browser-state) ---
        if not browser_state_mode and (token.startswith("<") or token == "<"):
            j = i
            tag_end = min(i + 20, n)
            while j < tag_end and ">" not in token_texts[j]:
                j += 1

            if j < n and ">" in token_texts[j]:
                tag_str = "".join(token_texts[i : j + 1])

                if "</" in tag_str:
                    closing_match = re.match(r".*?</(\w+)", tag_str)
                    if closing_match and len(stack) > 1:
                        closing_name = closing_match.group(1)
                        # Close any implicit tags (like <step>)
                        if closing_name == "agent_history":
                            for idx in range(len(stack) - 1, 0, -1):
                                if (
                                    stack[idx].node_type == "html_tag"
                                    and "step" in stack[idx].label
                                ):
                                    stack[idx].end_idx = i - 1
                                    while len(stack) > idx:
                                        stack.pop()
                                    break
                        for idx in range(len(stack) - 1, 0, -1):
                            if (
                                stack[idx].node_type == "html_tag"
                                and closing_name in stack[idx].label
                            ):
                                stack[idx].end_idx = j
                                while len(stack) > idx:
                                    stack.pop()
                                current_parent = stack[-1]
                                break
                else:
                    tag_match = re.match(r".*?<(\w+)", tag_str)
                    if tag_match:
                        tag_name = tag_match.group(1)

                        # Implicitly close previous <step>
                        if tag_name == "step":
                            for idx in range(len(stack) - 1, 0, -1):
                                if (
                                    stack[idx].node_type == "html_tag"
                                    and "step" in stack[idx].label
                                ):
                                    stack[idx].end_idx = i - 1
                                    while len(stack) > idx:
                                        stack.pop()
                                    current_parent = stack[-1]
                                    break

                        k = j + 1
                        if tag_name == "html" and tag_str.endswith("/>"):
                            # Collect inline text for <html /> tags
                            while k < n and token_texts[k] not in ["\n", "\t"]:
                                if re.match(
                                    r"^\*?\[\d+\]<", token_texts[k]
                                ) or token_texts[k].startswith("<"):
                                    break
                                k += 1

                        display = (
                            tag_str[:40]
                            if len(tag_str) < 40
                            else tag_str[:37] + "..."
                        )
                        node = TreeNode(
                            "html_tag",
                            display,
                            i,
                            k - 1 if tag_name == "html" and tag_str.endswith("/>") else j,
                        )
                        current_parent.children.append(node)

                        # <html /> becomes root of browser state hierarchy
                        if tag_name == "html" and tag_str.endswith("/>"):
                            browser_depth_stack = [
                                current_chatml_node
                                if current_chatml_node
                                else current_parent,
                                node,
                            ]
                            browser_state_mode = True
                            i = k
                            continue

                        if not tag_str.endswith("/>") and "/>" not in tag_str:
                            stack.append(node)
                            current_parent = node

                i = j + 1
                continue

        # Skip whitespace
        if not token.strip() or token == "\n":
            i += 1
            continue

        # --- Text content (non-browser-state) ---
        if not browser_state_mode:
            text_start = i
            text_parts = [token]
            i += 1
            while i < n:
                if token_texts[i].startswith("<") or token_texts[i] == "<":
                    break
                if "<|im" in token_texts[i]:
                    break
                if token_texts[i] == "\t":
                    break
                text_parts.append(token_texts[i])
                if "\n" in token_texts[i]:
                    i += 1
                    break
                i += 1

            text = "".join(text_parts).strip()
            if text:
                display = text[:47] + "..." if len(text) > 50 else text
                node = TreeNode("text", display, text_start, i - 1)
                current_parent.children.append(node)
        else:
            i += 1

    return root


def _fix_parent_ranges(node: TreeNode) -> None:
    """Fix parent node ranges to span from their start to their last child's end."""
    if node.children:
        for child in node.children:
            _fix_parent_ranges(child)
        last_child = node.children[-1]
        if last_child.end_idx > node.end_idx:
            node.end_idx = last_child.end_idx


def extract_leaf_chunks(
    root: TreeNode,
    min_chunk_size: int = 16,
    max_chunk_size: int = 256,
) -> List[FlatChunk]:
    """
    Walk the tree and extract all leaf nodes as FlatChunks.

    Small leaves (< min_chunk_size) are merged with the next sibling.
    Large leaves (> max_chunk_size) are split into sub-chunks.
    Returns sorted list of non-overlapping FlatChunk objects.
    """
    _fix_parent_ranges(root)
    raw_leaves = []
    _collect_leaves(root, raw_leaves)

    if not raw_leaves:
        return []

    # Sort by start index
    raw_leaves.sort(key=lambda n: n.start_idx)

    # Fill gaps: ensure all tokens are covered
    filled = []
    prev_end = 0
    for leaf in raw_leaves:
        if leaf.start_idx > prev_end:
            filled.append(
                TreeNode("gap", "gap", prev_end, leaf.start_idx - 1)
            )
        filled.append(leaf)
        prev_end = leaf.end_idx + 1
    # Cover trailing tokens
    if prev_end <= root.end_idx:
        filled.append(TreeNode("gap", "gap", prev_end, root.end_idx))

    # Process chunks: gap nodes (parent attribute tokens) become their own chunks
    # so they can be independently selected. Small leaves are merged with siblings,
    # large leaves are split.
    chunks = []
    chunk_id = 0
    merge_buf_start = None
    merge_buf_end = None
    merge_buf_label = None

    for leaf in filled:
        # Gap nodes (parent attribute tokens) always become their own chunk
        if leaf.node_type == "gap":
            # Flush any pending merge buffer first
            if merge_buf_start is not None:
                chunks.append(
                    FlatChunk(merge_buf_start, merge_buf_end, chunk_id, merge_buf_label)
                )
                chunk_id += 1
                merge_buf_start = None
            chunks.append(FlatChunk(leaf.start_idx, leaf.end_idx, chunk_id, leaf.label))
            chunk_id += 1
            continue

        if merge_buf_start is not None:
            # Try to extend the merge buffer
            combined_size = leaf.end_idx - merge_buf_start + 1
            if combined_size <= max_chunk_size:
                merge_buf_end = leaf.end_idx
                merge_buf_label += "+" + leaf.label
                if combined_size >= min_chunk_size:
                    # Flush
                    chunks.append(
                        FlatChunk(merge_buf_start, merge_buf_end, chunk_id, merge_buf_label)
                    )
                    chunk_id += 1
                    merge_buf_start = None
                continue
            else:
                # Flush the merge buffer as-is
                chunks.append(
                    FlatChunk(merge_buf_start, merge_buf_end, chunk_id, merge_buf_label)
                )
                chunk_id += 1
                merge_buf_start = None

        size = leaf.token_count
        if size < min_chunk_size:
            merge_buf_start = leaf.start_idx
            merge_buf_end = leaf.end_idx
            merge_buf_label = leaf.label
        elif size > max_chunk_size:
            # Split into sub-chunks
            for s in range(leaf.start_idx, leaf.end_idx + 1, max_chunk_size):
                e = min(s + max_chunk_size - 1, leaf.end_idx)
                chunks.append(FlatChunk(s, e, chunk_id, leaf.label))
                chunk_id += 1
        else:
            chunks.append(FlatChunk(leaf.start_idx, leaf.end_idx, chunk_id, leaf.label))
            chunk_id += 1

    # Flush remaining merge buffer
    if merge_buf_start is not None:
        chunks.append(
            FlatChunk(merge_buf_start, merge_buf_end, chunk_id, merge_buf_label)
        )
        chunk_id += 1

    # Re-number chunk IDs
    for i, c in enumerate(chunks):
        c.chunk_id = i

    return chunks


def _collect_leaves(node: TreeNode, leaves: List[TreeNode]):
    """Recursively collect leaf nodes."""
    if node.is_leaf:
        if node.token_count > 0:
            leaves.append(node)
    else:
        for child in node.children:
            _collect_leaves(child, leaves)


def make_fixed_chunks(
    seq_len: int, chunk_size: int = 64
) -> List[FlatChunk]:
    """Create fixed-size chunks as fallback when tree parsing isn't possible."""
    chunks = []
    for i in range(0, seq_len, chunk_size):
        end = min(i + chunk_size - 1, seq_len - 1)
        chunks.append(FlatChunk(i, end, len(chunks), f"chunk_{len(chunks)}"))
    return chunks


def format_tree(node: TreeNode, indent: int = 0, max_depth: int = 4) -> str:
    """Format a TreeNode as an indented string for logging."""
    if indent > max_depth:
        return ""
    prefix = "  " * indent
    label = node.label[:60] + "..." if len(node.label) > 60 else node.label
    line = f"{prefix}[{node.node_type}] {label} [{node.start_idx}-{node.end_idx}] ({node.token_count} tokens)"
    lines = [line]
    for child in node.children:
        child_str = format_tree(child, indent + 1, max_depth)
        if child_str:
            lines.append(child_str)
    if indent <= 1 and len(node.children) > 10:
        # Summarize if too many children at top levels
        lines = lines[:8]
        lines.append(f"{prefix}  ... ({len(node.children) - 7} more children)")
    return "\n".join(lines)


def format_chunks(chunks: List[FlatChunk]) -> str:
    """Format chunks as a compact summary for logging."""
    lines = []
    for c in chunks:
        label = c.label[:40] + "..." if len(c.label) > 40 else c.label
        lines.append(f"  chunk_{c.chunk_id}: [{c.start_idx}-{c.end_idx}] ({c.token_count} tokens) {label}")
    if len(lines) > 15:
        summary = lines[:7] + [f"  ... ({len(chunks) - 14} more chunks) ..."] + lines[-7:]
        return "\n".join(summary)
    return "\n".join(lines)


def build_chunks_from_token_ids(
    token_ids: List[int],
    tokenizer,
    min_chunk_size: int = 16,
    max_chunk_size: int = 256,
) -> List[FlatChunk]:
    """
    High-level entry point: decode token IDs to text, parse tree, extract chunks.

    Falls back to fixed-size chunking if tree parsing fails or produces no structure.
    """
    if not token_ids:
        return []

    try:
        # Decode each token individually to get per-token text
        token_texts = []
        for tid in token_ids:
            text = tokenizer.decode([tid])
            token_texts.append(text)

        # Parse tree
        tree = parse_chatml_tree(token_texts)

        logger.info(
            f"[TreeSparse] Parsed tree structure ({len(token_ids)} tokens):\n"
            f"{format_tree(tree)}"
        )

        # Extract leaf chunks
        chunks = extract_leaf_chunks(tree, min_chunk_size, max_chunk_size)

        if not chunks:
            logger.debug("Tree parsing produced no chunks, falling back to fixed-size")
            return make_fixed_chunks(len(token_ids))

        total_covered = sum(c.token_count for c in chunks)
        logger.info(
            f"[TreeSparse] Extracted {len(chunks)} chunks covering {total_covered}/{len(token_ids)} tokens:\n"
            f"{format_chunks(chunks)}"
        )

        return chunks

    except Exception as e:
        logger.warning(f"Tree parsing failed: {e}, falling back to fixed-size chunks")
        return make_fixed_chunks(len(token_ids))
