"""
structure_aware_chunker.py
--------------------------
Smart, structure-aware document chunking that respects semantic boundaries.

Design Philosophy:
- Paragraphs are atomic units — never split mid-paragraph if avoidable.
- Headers signal topic transitions — always chunk on them.
- Overlapping windows improve recall at retrieval time (configurable stride).
- Every chunk carries rich metadata for downstream explainability.

Why not LangChain's RecursiveCharacterTextSplitter?
  It is a great fallback, but it uses brute-force regex and doesn't understand
  document semantics (e.g., it may split a code block or numbered list mid-item).
  Our chunker builds a parse tree first, then decides where to cut.
"""

from __future__ import annotations

import re
import hashlib
import unicodedata
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator

# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

class BlockType(Enum):
    HEADING    = auto()
    PARAGRAPH  = auto()
    CODE       = auto()
    LIST       = auto()
    TABLE      = auto()
    SEPARATOR  = auto()


@dataclass
class Block:
    """A single semantic unit parsed from raw text."""
    block_type : BlockType
    text       : str
    heading_level: int = 0          # 1-6 for headings, 0 otherwise
    line_start : int   = 0          # 0-indexed line number in source


@dataclass(frozen=True)
class ChunkMetadata:
    """Immutable metadata pinned to every chunk — the explainability payload."""
    doc_id       : str              # e.g. "annual_report_2024.pdf"
    chunk_index  : int              # position in the ordered chunk list
    chunk_hash   : str              # SHA-256 of chunk text (dedup & cache key)
    heading_path : tuple[str, ...]  # breadcrumb: ("Section 2", "2.1 Results")
    page_hint    : int | None       # best-effort page number (PDF-derived)
    char_start   : int              # character offset in original document
    char_end     : int
    block_types  : tuple[str, ...]  # dominant block types inside this chunk


@dataclass
class Chunk:
    text     : str
    metadata : ChunkMetadata

    def __len__(self) -> int:
        return len(self.text)


# ──────────────────────────────────────────────────────────────
# Parser  (Markdown / plain text)
# ──────────────────────────────────────────────────────────────

_HEADING_RE   = re.compile(r"^(#{1,6})\s+(.*)")
_LIST_ITEM_RE = re.compile(r"^(\s*[-*+]|\s*\d+[.)]\s)")
_CODE_FENCE   = re.compile(r"^```|^~~~")
_TABLE_ROW_RE = re.compile(r"^\|.*\|")
_BLANK_RE     = re.compile(r"^\s*$")


def _classify_line(line: str, in_code_block: bool) -> BlockType:
    if in_code_block:
        return BlockType.CODE
    if _HEADING_RE.match(line):
        return BlockType.HEADING
    if _LIST_ITEM_RE.match(line):
        return BlockType.LIST
    if _TABLE_ROW_RE.match(line):
        return BlockType.TABLE
    if _BLANK_RE.match(line):
        return BlockType.SEPARATOR
    return BlockType.PARAGRAPH


def parse_blocks(text: str) -> list[Block]:
    """
    Convert raw text into a flat list of semantic Blocks.
    Handles Markdown syntax as well as plain prose.
    """
    lines      : list[str]  = text.splitlines()
    blocks     : list[Block] = []
    in_code    : bool        = False
    buf        : list[str]   = []
    buf_type   : BlockType   = BlockType.PARAGRAPH
    buf_start  : int         = 0

    def flush(end_line: int) -> None:
        nonlocal buf, buf_type, buf_start
        raw = "\n".join(buf).strip()
        if raw:
            hl = 0
            if buf_type == BlockType.HEADING:
                m  = _HEADING_RE.match(buf[0])
                hl = len(m.group(1)) if m else 0
            blocks.append(Block(buf_type, raw, hl, buf_start))
        buf       = []
        buf_type  = BlockType.PARAGRAPH
        buf_start = end_line + 1

    for i, line in enumerate(lines):
        # Toggle code fences
        if _CODE_FENCE.match(line):
            flush(i - 1)
            in_code = not in_code
            buf_type = BlockType.CODE
            buf_start = i
            continue

        ltype = _classify_line(line, in_code)

        if ltype == BlockType.SEPARATOR:
            flush(i - 1)
            continue

        # Headings always start a new block
        if ltype == BlockType.HEADING and buf:
            flush(i - 1)

        # Type change (e.g. paragraph → list) starts a new block
        if buf and ltype != buf_type and buf_type not in (BlockType.CODE,):
            flush(i - 1)

        if not buf:
            buf_type  = ltype
            buf_start = i
        buf.append(line)

    flush(len(lines) - 1)
    return blocks


# ──────────────────────────────────────────────────────────────
# Chunker
# ──────────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _char_offset(blocks: list[Block], up_to: int, original: str) -> int:
    """Estimate character offset by searching the original text."""
    if up_to == 0:
        return 0
    needle = blocks[up_to - 1].text[:40]
    idx    = original.rfind(needle)
    return idx + len(needle) if idx != -1 else 0


class StructureAwareChunker:
    """
    Chunks documents while respecting semantic structure.

    Parameters
    ----------
    max_tokens : int
        Soft upper limit on tokens per chunk (1 token ≈ 4 chars).
    overlap_tokens : int
        How many tokens of context to repeat at chunk boundaries.
        Overlap improves recall for questions that span chunk edges.
    respect_headings : bool
        If True, always start a new chunk when a heading is encountered.
    min_chunk_tokens : int
        Discard or merge chunks smaller than this to avoid noisy fragments.
    """

    def __init__(
        self,
        max_tokens       : int  = 400,
        overlap_tokens   : int  = 60,
        respect_headings : bool = True,
        min_chunk_tokens : int  = 40,
    ) -> None:
        self.max_chars     = max_tokens      * 4
        self.overlap_chars = overlap_tokens  * 4
        self.min_chars     = min_chunk_tokens * 4
        self.respect_headings = respect_headings

    # ── public API ─────────────────────────────────────────────

    def chunk(
        self,
        text    : str,
        doc_id  : str,
        page_map: dict[int, int] | None = None,  # char_offset → page_number
    ) -> list[Chunk]:
        """
        Primary entry point. Returns an ordered list of Chunks.

        page_map is an optional dict mapping character offsets to page numbers
        (extracted by pdf_parser.py). When provided, each chunk carries a
        page_hint for display in the UI.
        """
        text   = self._normalize(text)
        blocks = parse_blocks(text)
        return list(self._blocks_to_chunks(blocks, text, doc_id, page_map or {}))

    # ── internals ──────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        # Unicode NFC normalization + collapse excessive blank lines
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _blocks_to_chunks(
        self,
        blocks   : list[Block],
        original : str,
        doc_id   : str,
        page_map : dict[int, int],
    ) -> Iterator[Chunk]:
        """
        Walk blocks, accumulate text until max_chars, then emit a Chunk.
        Heading blocks act as mandatory cut-points when respect_headings=True.
        """
        buf_blocks  : list[Block] = []
        buf_chars   : int         = 0
        heading_path: list[str]   = []
        chunk_idx   : int         = 0
        char_cursor : int         = 0

        def emit(overlap_text: str = "") -> Iterator[Chunk]:
            nonlocal chunk_idx, char_cursor
            if not buf_blocks:
                return

            full_text = "\n\n".join(b.text for b in buf_blocks)
            if overlap_text:
                full_text = overlap_text + "\n\n" + full_text

            if len(full_text) < self.min_chars:
                return  # too small — will be merged into next chunk

            char_start = char_cursor
            char_end   = char_start + len(full_text)
            page_hint  = self._lookup_page(char_start, page_map)

            yield Chunk(
                text = full_text,
                metadata = ChunkMetadata(
                    doc_id       = doc_id,
                    chunk_index  = chunk_idx,
                    chunk_hash   = _sha256(full_text),
                    heading_path = tuple(heading_path),
                    page_hint    = page_hint,
                    char_start   = char_start,
                    char_end     = char_end,
                    block_types  = tuple(b.block_type.name for b in buf_blocks),
                ),
            )
            chunk_idx  += 1
            char_cursor = char_end

        overlap_carry = ""

        for block in blocks:
            is_heading     = block.block_type == BlockType.HEADING
            would_overflow = (buf_chars + len(block.text)) > self.max_chars

            # Mandatory cut on heading
            if buf_blocks and self.respect_headings and is_heading:
                yield from emit(overlap_carry)
                overlap_carry = self._extract_overlap(buf_blocks)
                buf_blocks, buf_chars = [], 0

            # Soft cut on overflow
            elif would_overflow and buf_blocks:
                yield from emit(overlap_carry)
                overlap_carry = self._extract_overlap(buf_blocks)
                buf_blocks, buf_chars = [], 0

            # Track heading breadcrumbs
            if is_heading:
                level = block.heading_level
                heading_path = heading_path[:level - 1] + [block.text]

            buf_blocks.append(block)
            buf_chars  += len(block.text)

        yield from emit(overlap_carry)   # flush remainder

    def _extract_overlap(self, buf_blocks: list[Block]) -> str:
        """
        Take the tail of the current buffer as overlap context for the next chunk.
        We walk blocks from the end until we've collected ~overlap_chars.
        """
        collected, total = [], 0
        for block in reversed(buf_blocks):
            total += len(block.text)
            collected.insert(0, block.text)
            if total >= self.overlap_chars:
                break
        return "\n\n".join(collected)

    @staticmethod
    def _lookup_page(char_offset: int, page_map: dict[int, int]) -> int | None:
        if not page_map:
            return None
        best = None
        for offset, page in sorted(page_map.items()):
            if offset <= char_offset:
                best = page
            else:
                break
        return best
