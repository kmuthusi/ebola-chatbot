"""Ollama embeddings, Chroma persistence, and freshness-aware querying."""

import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import ollama
from dotenv import load_dotenv

load_dotenv()
DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"
INDEX_SCHEMA = "pdf-directory-v3-structure-aware"
_SEARCH_STOPWORDS = {"a", "about", "an", "and", "are", "can", "do", "does", "for", "how", "in", "is", "of", "or", "the", "there", "to", "what", "when", "where", "which", "who", "with"}


def _search_words(text: str) -> list[str]:
    words = re.findall(r"[a-z]{2,}", text.lower())
    return [word[:-1] if len(word) > 5 and word.endswith("s") and not word.endswith(("us", "ss")) else word for word in words]


def _is_publication_matter(text: str, query_terms: set[str]) -> bool:
    if query_terms & {"author", "copyright", "isbn", "license", "publication", "publisher", "title"}:
        return False
    lower = text.lower()
    return (len(text) < 100 and "." not in text) or any(
        marker in lower for marker in ("some rights reserved", "suggested citation", "all rights reserved", "isbn 978")
    )


def _hash_manifest(items: List[Dict[str, Any]]) -> str:
    """Hash source content, not just page lengths, so edits trigger reindexing."""
    digest = hashlib.sha256()
    for item in items:
        digest.update(str(item.get("source", "")).encode())
        digest.update(str(item.get("page")).encode())
        digest.update(str(item.get("text", "")).encode())
    digest.update(INDEX_SCHEMA.encode())
    return digest.hexdigest()


def _join_overlapping_chunks(chunks: List[str]) -> str:
    if not chunks:
        return ""
    joined = chunks[0]
    for chunk in chunks[1:]:
        overlap = next((size for size in range(min(250, len(joined), len(chunk)), 29, -1)
                        if joined.endswith(chunk[:size])), 0)
        joined += chunk[overlap:] if overlap else "\n" + chunk
    return joined


class VectorStore:
    def __init__(
        self,
        collection_name: str = "ebola_sources_ollama",
        chroma_dir: str = "chroma_db",
        embed_model: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.embed_model = embed_model or os.getenv(
            "OLLAMA_EMBED_MODEL", DEFAULT_EMBED_MODEL
        )
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(collection_name)
        self.manifest_path = self.chroma_dir / f"{collection_name}.manifest.json"
        self._manifest_hash: Optional[str] = None
        self._source_fingerprint: Optional[str] = None
        self._lexical_cache = None
        self._query_cache: Dict[tuple[str, int, str | None], List[Dict[str, Any]]] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text())
                self._manifest_hash = manifest.get("hash")
                if manifest.get("embed_model") == self.embed_model and manifest.get("index_schema") == INDEX_SCHEMA:
                    self._source_fingerprint = manifest.get("source_fingerprint")
            except (OSError, json.JSONDecodeError):
                self._manifest_hash = None

    def source_is_current(self, fingerprint: str) -> bool:
        return self._source_fingerprint == fingerprint and self.collection.count() > 0

    def mark_source(self, fingerprint: str) -> None:
        self._source_fingerprint = fingerprint
        self.manifest_path.write_text(json.dumps({
            "hash": self._manifest_hash,
            "embed_model": self.embed_model,
            "index_schema": INDEX_SCHEMA,
            "source_fingerprint": fingerprint,
        }, indent=2))

    def _chunk_text(self, text: str, chunk_len: int = 1200, overlap: int = 200) -> List[str]:
        if chunk_len <= 0 or overlap < 0 or overlap >= chunk_len:
            raise ValueError("chunk_len must be positive and overlap must be smaller than chunk_len")
        chunks = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_len, length)
            if end < length:
                lower = start + int(chunk_len * 0.60)
                for boundary in ("\n\n", "\n", ". ", "; "):
                    cut = text.rfind(boundary, lower, end)
                    if cut >= 0:
                        end = cut + len(boundary)
                        break
            chunks.append(text[start:end])
            if end == length:
                break
            start = max(start + 1, end - overlap)
        return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 40]

    def index_chunks(self, raw_chunks: List[Dict[str, Any]]) -> None:
        if not raw_chunks:
            raise ValueError("Cannot index an empty source.")
        manifest = hashlib.sha256((self.embed_model + _hash_manifest(raw_chunks)).encode()).hexdigest()

        batch_size = 32
        records = []
        for source_index, raw in enumerate(raw_chunks):
            for sub_index, text in enumerate(self._chunk_text(raw["text"])):
                records.append((source_index, sub_index, text, raw.get("page"), raw.get("source", "unknown.pdf"), raw.get("section", "body")))

        if not records:
            raise ValueError("No substantial text chunks found in the source PDFs.")

        try:
            self.client.delete_collection(self.collection_name)
        except ValueError:
            pass
        self.collection = self.client.get_or_create_collection(self.collection_name)
        self._lexical_cache = None
        self._query_cache.clear()

        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            embeddings = ollama.embed(
                model=self.embed_model,
                input=[record[2] for record in batch],
            )["embeddings"]
            self.collection.add(
                ids=[
                    f"chunk-{source}-{sub}-{hashlib.sha1(text.encode()).hexdigest()[:12]}"
                    for source, sub, text, _, _, _ in batch
                ],
                documents=[record[2] for record in batch],
                metadatas=[{"page": record[3], "source": record[4], "section": record[5]} for record in batch],
                embeddings=embeddings,
            )

        self._manifest_hash = manifest
        self._source_fingerprint = None
        self.manifest_path.write_text(json.dumps({"hash": manifest, "embed_model": self.embed_model,
                                                  "index_schema": INDEX_SCHEMA}, indent=2))

    def ensure_index(self, raw_chunks: List[Dict[str, Any]]) -> None:
        manifest = hashlib.sha256((self.embed_model + _hash_manifest(raw_chunks)).encode()).hexdigest()
        if self._manifest_hash != manifest or self.collection.count() == 0:
            self.index_chunks(raw_chunks)

    def query(self, query: str, k: int = 5, include_scores: bool = True, source: str | None = None) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        cache_key = (query, k, source)
        if include_scores and cache_key in self._query_cache:
            return [dict(item) for item in self._query_cache[cache_key]]
        query_embedding = ollama.embed(model=self.embed_model, input=query)["embeddings"][0]
        include = ["documents", "metadatas", "distances"] if include_scores else ["documents", "metadatas"]
        where = {"section": "body"}
        if source:
            where = {"$and": [{"section": "body"}, {"source": source}]}
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            where=where,
            include=include,
        )
        documents = result.get("documents", [[]])[0]
        metadata = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0] if include_scores else [None] * len(documents)
        output = []
        for document, meta, distance in zip(documents, metadata, distances):
            output.append(
                {
                    "text": document,
                    "page": meta.get("page"),
                    "source": meta.get("source"),
                    "section": meta.get("section"),
                    "score": (1.0 / (1.0 + distance)) if distance is not None else None,
                }
            )
        if include_scores:
            self._query_cache[cache_key] = [dict(item) for item in output]
        return output

    def keyword_query(self, phrase: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve exact source passages when embedding search misses key wording."""
        result = self.collection.get(
            where={"section": "body"},
            where_document={"$contains": phrase},
            include=["documents", "metadatas"],
            limit=k,
        )
        return [
            {"text": text, "page": meta.get("page"), "source": meta.get("source"),
             "section": meta.get("section"), "score": 0.9}
            for text, meta in zip(result["documents"], result["metadatas"])
        ]

    def _lexical_records(self):
        if self._lexical_cache is None:
            result = self.collection.get(
                where={"section": "body"}, include=["documents", "metadatas"]
            )
            records = []
            document_frequency = Counter()
            total_length = 0
            for text, meta in zip(result["documents"], result["metadatas"]):
                words = _search_words(text)
                counts = Counter(words)
                records.append((text, meta, counts, len(words)))
                document_frequency.update(counts.keys())
                total_length += len(words)
            average_length = total_length / max(len(records), 1)
            self._lexical_cache = (records, document_frequency, average_length)
        return self._lexical_cache

    def expand_evidence(self, evidence: List[Dict[str, Any]], max_chars: int = 1600,
                        question: str = "") -> List[Dict[str, Any]]:
        """Restore nearby text on the same cited page without another embedding call."""
        records, document_frequency, _ = self._lexical_records()
        generic_modifiers = {"extra", "specific", "specify", "guideline", "guidance", "document", "source"}
        focus_terms = [term for term in set(_search_words(question)) - _SEARCH_STOPWORDS - generic_modifiers
                       if document_frequency[term]]
        focus_terms.sort(key=lambda term: document_frequency[term])
        by_page = {}
        for text, meta, _, _ in records:
            if meta.get("section") == "body":
                by_page.setdefault((meta.get("source"), meta.get("page")), []).append(text)
        expanded = []
        for hit in evidence:
            if hit.get("is_definition") or hit.get("is_focused"):
                expanded.append(hit)
                continue
            chunks = by_page.get((hit.get("source"), hit.get("page")), [])
            if len(chunks) < 2:
                expanded.append(hit)
                continue
            try:
                position = chunks.index(hit["text"])
            except ValueError:
                expanded.append(hit)
                continue
            full_page = _join_overlapping_chunks(chunks)
            for term in focus_terms:
                match = re.search(rf"\b{re.escape(term)}\w*", full_page, re.I)
                if match:
                    start = max(0, match.start() - 500)
                    expanded.append({**hit, "text": full_page[start:start + max_chars]})
                    break
            else:
                center = chunks[position][:max_chars]
                spare = max(0, max_chars - len(center) - 2)
                before = chunks[position - 1][-spare // 2:] if position and spare else ""
                after = chunks[position + 1][:spare - len(before)] if position + 1 < len(chunks) and spare else ""
                expanded.append({**hit, "text": "\n".join(part for part in (before, center, after) if part)})
        return expanded

    def lexical_search(self, question: str, k: int = 8) -> List[Dict[str, Any]]:
        """Cheap second-chance search over cached chunks, with no model call."""
        records, document_frequency, average_length = self._lexical_records()
        terms = set(_search_words(question)) - _SEARCH_STOPWORDS
        scored = []
        for text, meta, counts, length in records:
            if meta.get("section") != "body" or _is_publication_matter(text, terms):
                continue
            score = sum(
                math.log(1 + (len(records) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                * counts[term] * 2.2 / (counts[term] + 1.2 * (0.25 + 0.75 * length / max(average_length, 1)))
                for term in terms if counts[term]
            )
            if score:
                scored.append((score, text, meta))
        scored.sort(key=lambda row: row[0], reverse=True)
        return [
            {"text": text, "page": meta.get("page"), "source": meta.get("source"),
             "section": "body", "score": min(0.60, 0.35 + 0.02 * score)}
            for score, text, meta in scored[:k]
        ]

    def definition_search(self, subject: str, k: int = 2) -> List[Dict[str, Any]]:
        """Find sentences that actually define a subject, regardless of domain."""
        records, _, _ = self._lexical_records()
        if not subject.strip():
            return []
        topic = re.compile(rf"\b{re.escape(subject.casefold())}\b")
        matches = []
        for text, meta, _, _ in records:
            if meta.get("section") != "body" or _is_publication_matter(text, {subject}):
                continue
            compact = re.sub(r"\s+", " ", text)
            for sentence in re.split(r"(?<=[.!?])\s+", compact):
                lower_sentence = sentence.casefold()
                definition = re.search(r"\b(?:is an?|are an?|refers to|means|defined as|is caused by|are caused by)\b", lower_sentence)
                mentions = [match for match in topic.finditer(lower_sentence)
                            if definition and match.start() < definition.start()]
                mention = mentions[-1] if mentions else None
                max_prefix = 250 if len(subject.split()) > 1 else 12
                if not mention or mention.start() > max_prefix or len(sentence) > 380 or len(sentence) < 30:
                    continue
                if re.search(r"\b(?:diagnostic testing for|compared with previous|outbreak caused by|study of|references)\b", lower_sentence):
                    continue
                if not definition:
                    continue
                distance = abs(definition.start() - mention.end())
                if distance > 180:
                    continue
                score = 1.0 / (1 + distance / 30) + (0.7 if mention.start() < 30 else 0)
                excerpt = sentence[mention.start():] if len(subject.split()) > 1 else sentence
                matches.append((score, excerpt, meta))
                break
        matches.sort(key=lambda item: item[0], reverse=True)
        seen = set()
        output = []
        for _, text, meta in matches:
            key = (meta.get("source"), meta.get("page"))
            if key in seen:
                continue
            seen.add(key)
            output.append({"text": text, "page": key[1], "source": key[0], "section": "body", "score": .9})
            if len(output) >= k:
                break
        return output

    def search(self, question: str, k: int = 30) -> List[Dict[str, Any]]:
        """Combine semantic and lexical matches from every indexed PDF."""
        semantic = self.query(question, k=max(k, 40))
        records, document_frequency, average_length = self._lexical_records()
        terms = set(_search_words(question)) - _SEARCH_STOPWORDS
        if len(terms) >= 5:
            focus = [term for term in sorted(terms, key=lambda term: (document_frequency[term] or len(records), term))
                     if document_frequency[term]][:3]
            if focus:
                semantic.extend(self.query(" ".join(focus), k=20))
        lexical = []
        for text, meta, counts, length in records:
            score = 0.0
            for term in terms:
                frequency = counts[term]
                if frequency:
                    idf = math.log(1 + (len(records) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                    score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / max(average_length, 1)))
            if score and not _is_publication_matter(text, terms):
                lexical.append((score, text, meta))
        lexical.sort(key=lambda item: item[0], reverse=True)

        merged = {}
        for hit in semantic:
            if not _is_publication_matter(hit["text"], terms):
                merged[(hit["source"], hit["page"], hit["text"])] = hit
        for score, text, meta in lexical[:max(k, 20)]:
            key = (meta.get("source"), meta.get("page"), text)
            lexical_score = min(0.60, 0.35 + 0.02 * score)
            if key in merged:
                merged[key]["score"] = max(merged[key]["score"] or 0, lexical_score)
            else:
                merged[key] = {
                    "text": text, "page": meta.get("page"), "source": meta.get("source"),
                    "section": meta.get("section"), "score": lexical_score,
                }
        return list(merged.values())
