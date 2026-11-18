"""Step 1: turn the official XML pages into clean text chunks.

Input:  data/raw/vosdroits/F*.xml
        The "Vos droits et démarches - Particuliers" dataset from DILA
        (Service-Public.gouv.fr), Licence Ouverte / Open Licence 2.0.
Output: data/corpus/chunks.jsonl, one JSON object per chunk (committed to git,
        so every later step is reproducible even when the dataset changes).

Why structure-aware chunking?
The pages contain "cases": the same question can have one answer for an
unfurnished flat and another for a furnished flat. If we cut the text every
N characters, a chunk could lose the case it belongs to. So we cut along the
document's own sections and start every chunk with its heading path, e.g.

    Dépôt de garantie dans un bail d'habitation > Logement meublé >
    À quelle condition le dépôt de garantie est restitué ? > État des lieux ...

Run:  python -m rag_assistant.ingest
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_assistant.config import CHUNKS_PATH, RAW_DIR

DC = "{http://purl.org/dc/elements/1.1/}"

# The slice of the corpus we keep: renting and work (about 360 pages).
KEEP_SUBTHEMES = {
    "Bail d’habitation",
    "Aides financières pour le logement",
    "Congés",
    "Quitter son emploi",
    "Contrats et carrière",
}
# "Recherche guidée" pages are interactive question trees, not readable text.
SKIP_TYPES = {"Recherche guidée"}

# Elements that start a new section. Their <Titre> becomes part of the path.
SECTION_TAGS = {"Situation", "Chapitre", "SousChapitre", "Cas"}
# Elements we ignore completely (metadata, duplicates, navigation).
IGNORE_TAGS = {"Condition", "TitreRiche", "Titre", "Source", "Reference", "Video", "Credits"}
# Boxes that keep their own label in the text.
LABELLED_BOXES = {"ANoter": "À noter", "ASavoir": "À savoir", "Attention": "Attention",
                  "Rappel": "Rappel", "Exemple": "Exemple"}

MAX_CHARS = 1200  # body size limit; the e5 model reads up to 512 tokens (~1800 chars)


@dataclass
class Chunk:
    chunk_id: str
    page_id: str
    page_title: str
    url: str
    section: str  # heading path inside the page
    text: str  # what gets embedded: title + path + body
    modified: str


def clean(text: str) -> str:
    """Collapse whitespace and non-breaking spaces into single spaces."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def inline_text(el: ET.Element) -> str:
    return clean(" ".join(el.itertext()))


def title_of(el: ET.Element) -> str:
    """Text of the element's own <Titre> (which may wrap a <Paragraphe>)."""
    t = el.find("Titre")
    return inline_text(t) if t is not None else ""


def render_block(el: ET.Element) -> list[str]:
    """Render one content element (paragraph, list, table...) as lines of text."""
    tag = el.tag
    if tag in IGNORE_TAGS:
        return []
    if tag == "Paragraphe":
        t = inline_text(el)
        return [t] if t else []
    if tag == "Liste":
        return ["- " + inline_text(item) for item in el.findall("Item") if inline_text(item)]
    if tag == "Tableau":
        rows = []
        for row in el.iter("Rangée"):
            cells = [inline_text(c) for c in row.findall("Cellule")]
            rows.append(" | ".join(cells))
        title = title_of(el)
        return ([title] if title else []) + rows
    if tag == "TitreFlottant":  # a small sub-heading inside a section
        t = inline_text(el)
        return [f"## {t}"] if t else []
    if tag in LABELLED_BOXES:
        inner = [line for child in el for line in render_block(child)]
        return [f"{LABELLED_BOXES[tag]} : " + " ".join(inner)] if inner else []
    if tag in ("ServiceEnLigne", "OuSAdresser"):
        title = title_of(el)
        return [f"[Service : {title}]"] if title else []
    # Unknown wrapper (Texte, Introduction, BlocCas, ...): render its children.
    return [line for child in el for line in render_block(child)]


# Containers with no title of their own: we look straight through them.
TRANSPARENT_TAGS = {"Texte", "BlocCas", "ListeSituations", "Introduction"}

Section = tuple[list[str], list[str]]  # (heading path, lines of text)


def walk(el: ET.Element, path: list[str], out: list[Section]) -> None:
    """Depth-first walk: open a new section for `el`, then visit its children."""
    current: list[str] = []
    out.append((path, current))
    visit_children(el, path, current, out)


def visit_children(el: ET.Element, path: list[str], current: list[str], out: list[Section]) -> None:
    for child in el:
        if child.tag in IGNORE_TAGS:
            continue
        if child.tag in SECTION_TAGS:  # a titled sub-section: go one level deeper
            title = title_of(child)
            walk(child, path + [title] if title else path, out)
        elif child.tag in TRANSPARENT_TAGS:  # same level, same section
            visit_children(child, path, current, out)
        else:  # real content: paragraph, list, table, box...
            current.extend(render_block(child))


def split_lines(lines: list[str], max_chars: int = MAX_CHARS) -> list[list[str]]:
    """Group lines into pieces of at most max_chars, never cutting a line."""
    pieces, piece, size = [], [], 0
    for line in lines:
        if piece and size + len(line) > max_chars:
            pieces.append(piece)
            piece, size = [], 0
        piece.append(line)
        size += len(line) + 1
    if piece:
        pieces.append(piece)
    return pieces


def parse_page(xml_path: Path) -> list[Chunk] | None:
    root = ET.parse(xml_path).getroot()
    sub = root.find("SousThemePere")
    if sub is None or sub.text not in KEEP_SUBTHEMES or root.get("type") in SKIP_TYPES:
        return None

    page_id = root.get("ID")
    title = clean(root.findtext(f"{DC}title") or "")
    url = root.get("spUrl") or f"https://www.service-public.gouv.fr/particuliers/vosdroits/{page_id}"
    modified = (root.findtext(f"{DC}date") or "").replace("modified ", "")
    description = clean(root.findtext(f"{DC}description") or "")

    sections: list[Section] = []
    intro = root.find("Introduction")
    intro_lines = render_block(intro) if intro is not None else []
    sections.append(([], ([description] if description else []) + intro_lines))
    if root.find("Texte") is not None:
        walk(root.find("Texte"), [], sections)
    if root.find("ListeSituations") is not None:
        walk(root.find("ListeSituations"), [], sections)

    chunks: list[Chunk] = []
    for path, lines in sections:
        lines = [line for line in lines if line]
        if not lines:
            continue
        section = " > ".join(path)
        header = title + (f" > {section}" if section else "")
        for piece in split_lines(lines):
            body = "\n".join(piece)
            chunks.append(Chunk(
                chunk_id=f"{page_id}#{len(chunks)}",
                page_id=page_id, page_title=title, url=url, section=section,
                text=f"{header}\n{body}", modified=modified,
            ))
    return chunks


def main() -> None:
    files = sorted(RAW_DIR.glob("F*.xml"))
    if not files:
        raise SystemExit(f"No XML files in {RAW_DIR}. See README: 'Get the data'.")
    all_chunks, pages = [], 0
    for f in files:
        chunks = parse_page(f)
        if chunks:
            pages += 1
            all_chunks.extend(chunks)
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"{pages} pages -> {len(all_chunks)} chunks -> {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
