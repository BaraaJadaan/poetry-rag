import sys
sys.stdout.reconfigure(encoding="utf-8")

"""
preprocess.py — Ashaar dataset cleaning and chunking pipeline

Design decisions (interview-ready reasoning):
─────────────────────────────────────────────
1. Two text representations per chunk:
   - `text_display`: original text with tashkeel (diacritics) intact → shown in UI
   - `text_index`:   normalized text → used for embedding and BM25 indexing
   Rationale: stripping diacritics from the *display* copy degrades the reading
   experience for classical poetry. Stripping from the *index* copy ensures that
   a user who types without diacritics (the common case on a keyboard) still
   matches corpus verses diacriticized in the original source.

2. Hemistich pairing (صدر + عجز → بيت):
   The dataset stores each half-verse (شطر) as a separate list entry.
   Consecutive pairs form one complete verse (بيت). Odd-length lists (9.57% of
   poems) get their trailing orphan hemistich kept as a single-hemistich chunk
   rather than discarded — discarding would silently lose real content.

3. Metadata kept sparse:
   - poem description (94.3% null) → dropped entirely
   - poem title (36% null) → imputed from verse[0] first 40 chars
   - poem theme, poet location, poet description (>72% null) → kept in metadata
     as optional filter fields but never required by retrieval logic

4. Normalization applied (Unicode-safe, no external library needed):
   - Remove tashkeel (U+0610–U+061A, U+064B–U+065F, U+0670)
   - Normalize alef variants (أ إ آ ٱ → ا)
   - Normalize teh marbuta (ة → ه)
   - Normalize alef maqsura (ى → ي)
   These four are the standard set for Arabic IR; they reduce vocabulary
   fragmentation without losing semantic content.
"""

import re
import unicodedata
from datasets import load_dataset


# ── Arabic text normalization ─────────────────────────────────────────────────

# Unicode ranges for Arabic diacritics (harakat + shadda + sukun etc.)
_DIACRITIC_PATTERN = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670]"
)

# Alef with hamza above, hamza below, madda above, wasla
_ALEF_VARIANTS = str.maketrans("أإآٱ", "اااا")


def normalize_arabic(text: str) -> str:
    """
    Return a retrieval-normalized copy of an Arabic string.

    Transformations (in order):
      1. Strip tashkeel (diacritics) — fatha, kasra, damma, tanwin, shadda, sukun
      2. Normalize alef variants → bare alef (ا)
      3. Normalize teh marbuta → heh (ة → ه)
      4. Normalize alef maqsura → yeh (ى → ي)
      5. Collapse runs of whitespace

    This function is *not* applied to display text — only to the copy that
    goes into the vector store and BM25 index.
    """
    if not text:
        return text
    text = _DIACRITIC_PATTERN.sub("", text)   # step 1
    text = text.translate(_ALEF_VARIANTS)       # step 2
    text = text.replace("ة", "ه")              # step 3
    text = text.replace("ى", "ي")              # step 4
    text = " ".join(text.split())               # step 5
    return text


# ── Row-level validation ──────────────────────────────────────────────────────

def is_valid_row(row: dict) -> bool:
    """Return True only if this row should enter the pipeline at all."""
    verses = row.get("poem verses") or []
    # A poem with zero hemistichs has no content to embed.
    return len(verses) > 0


# ── Title imputation ──────────────────────────────────────────────────────────

def impute_title(row: dict) -> str:
    """
    Return the poem title, falling back to the first 40 chars of verse[0].

    Why 40 chars? Long enough to be recognizable in the UI; short enough to
    fit in a citation line without truncation on mobile.
    """
    title = row.get("poem title")
    if title and title.strip():
        return title.strip()
    verses = row.get("poem verses") or []
    if verses:
        return verses[0][:40].strip() + "…"
    return "بيت مجهول"  # fallback: "unknown verse"


# ── Hemistich pairing ─────────────────────────────────────────────────────────

def pair_hemistichs(verses: list[str]) -> list[tuple[str, str | None]]:
    """
    Convert a flat list of hemistichs into (sadr, ajuz) pairs.

    For even-length lists:  [(v[0], v[1]), (v[2], v[3]), ...]
    For odd-length lists:   same as above, trailing hemistich → (v[-1], None)

    The None sentinel in position 1 means "orphan hemistich" — the chunking
    function downstream decides how to render it (typically displayed as a
    single line rather than a paired verse).

    Why keep orphans instead of discarding them?
    Discarding silently loses content that was in the original source. Some
    orphans are real verses from poems that lost one hemistich to transcription
    error; others might be single-hemistich lines from non-classical forms.
    Either way, they carry semantic content worth embedding.
    """
    pairs = []
    i = 0
    while i < len(verses):
        sadr = verses[i]
        ajuz = verses[i + 1] if i + 1 < len(verses) else None
        pairs.append((sadr, ajuz))
        i += 2
    return pairs


# ── Chunk builder ─────────────────────────────────────────────────────────────

def build_chunks(row: dict) -> list[dict]:
    """
    Convert one dataset row into a list of chunk dicts ready for embedding.

    Each chunk represents one verse (بيت) and contains:
      - text_display:   original text (with diacritics) for UI rendering
      - text_index:     normalized text for embedding + BM25
      - bait_index:     position of this verse within its poem (0-based)
      - is_orphan:      True if this chunk has only one hemistich
      - poem_title:     display title (imputed if missing)
      - poet_name:      str or None
      - poet_era:       str or None
      - poem_meter:     str or None
      - poem_theme:     str or None
      - poem_url:       str or None
      - poet_url:       str or None
      - poem_language_type: str or None  ← used to filter classical vs. colloquial

    Fields deliberately excluded from chunk metadata:
      - poem description (94.3% null — not worth the storage or confusion)
      - poet description (>72% null — same reasoning)
      - poet location (>72% null — same reasoning)
    """
    verses = row.get("poem verses") or []
    pairs = pair_hemistichs(verses)
    title = impute_title(row)

    chunks = []
    for bait_idx, (sadr, ajuz) in enumerate(pairs):
        # Build the full-verse string for display (keeps *** as visual separator)
        if ajuz is not None:
            display = f"{sadr} *** {ajuz}"
            # For indexing: we need to handle words that were split mid-syllable across the
            # hemistich boundary (e.g. "الخَل" + "قَ"). If we only join with a space, 
            # BM25 will index "الخل" and "ق" as separate words, breaking keyword search for "الخلق".
            # FIX: We include BOTH the spaced version and the unspaced version in the index string.
            # If the word was split, `sadr+ajuz` reunites it perfectly for BM25. If it wasn't split, 
            # `sadr+ajuz` creates a nonsense compound word that BM25 will just index harmlessly 
            # as a rare token nobody will ever search for.
            raw_index = f"{sadr} {ajuz} {sadr}{ajuz}"
        else:
            display = sadr  # orphan hemistich — one half-line
            raw_index = sadr

        index_text = normalize_arabic(raw_index)

        chunk = {
            # ── Text fields ──────────────────────────────────────────────────
            "text_display": display,
            "text_index": index_text,
            # ── Structural metadata ───────────────────────────────────────────
            "bait_index": bait_idx,
            "is_orphan": ajuz is None,
            # ── Poem-level metadata ───────────────────────────────────────────
            "poem_title": title,
            "poet_name": row.get("poet name"),
            "poet_era": row.get("poet era"),
            "poem_meter": row.get("poem meter"),
            "poem_theme": row.get("poem theme"),
            "poem_url": row.get("poem url"),
            "poet_url": row.get("poet url"),
            "poem_language_type": row.get("poem language type"),
        }
        chunks.append(chunk)

    return chunks


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    dataset_name: str = "arbml/ashaar",
    split: str = "train",
    limit: int | None = None,
    classical_only: bool = False,
    poet_whitelist: set[str] | None = None,
    quality_only: bool = False,
) -> list[dict]:
    """
    Load the Ashaar dataset, clean it, and return a flat list of verse chunks.

    Args:
        dataset_name:   HuggingFace dataset identifier
        split:          which split to load (only 'train' exists in ashaar)
        limit:          if set, process only the first N rows (for quick tests)
        classical_only: if True, drop rows where poem_language_type is not
                        classical Arabic. Set this once you've inspected the
                        actual values of that field — see OPEN QUESTION below.
        poet_whitelist: if set, keep only poems whose poet name is in this set.
                        For the cloud deployment this curates the corpus down to
                        the canon of poets users actually quote (~10% of rows),
                        which makes embedding time, storage, and query latency
                        manageable on a 1 GB free-tier VM.
        quality_only:   if True, keep only complete verses (بيت with both
                        hemistichs) whose normalized text has no "..." ellipsis
                        marker. The preprocessing already merges most junk at
                        the sadr+ajuz join; this removes the residual noise
                        class that would otherwise pollute BM25 and the vector
                        store.

    Returns:
        List of chunk dicts, one per verse (بيت), ready for embedding.

    When poet_whitelist or quality_only is set, the dataset is scanned as a
    stream (row by row, bounded memory) and `limit` counts *matching poems*
    rather than raw rows — otherwise the first N rows of the file would almost
    never contain whitelisted poets and the smoke test would embed nothing.
    This streaming scan is also the foundation of the bounded-memory full
    indexer planned for the server (see DEPLOYMENT_GUIDE.md §10).

    OPEN QUESTION (resolve before full corpus embed):
        What are the actual values in `poem language type`?
        Run:  Counter(ds["poem language type"]).most_common(10)
        Then decide: is "فصيح" the only classical tag, or are there others?
        This determines what string to filter on in `classical_only` mode.
    """
    print(f"Loading {dataset_name} [{split}] from cache...")
    ds = load_dataset(dataset_name)[split]

    # The exclusion list is used by both the batch and the streaming path.
    # It is safer to exclude known colloquial tags than require classical tags,
    # so we don't drop the 28% that are unlabelled (which are overwhelmingly classical).
    exclusion_list = {"عامي", "شعبي", "-"}

    streaming = bool(poet_whitelist or quality_only)

    if limit and not streaming:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"  Limited to {limit} rows for testing.")

    print(f"  Loaded {len(ds):,} rows.")

    if not streaming:
        # ── Batch path (unchanged behaviour: local/full-corpus embedding) ──
        before = len(ds)
        ds = ds.filter(is_valid_row)
        after = len(ds)
        print(f"  Dropped {before - after:,} empty-verse rows -> {after:,} remaining.")

        if classical_only:
            before = len(ds)
            # The dataset uses multiple tags.
            # Keep: 'فصيح' (60.4%), 'None' (28.0%), 'فصحى' (8.2%)
            # Drop: 'عامي' (3.3%), 'شعبي' (0.1%), '-' (0.0%)
            ds = ds.filter(lambda r: r.get("poem language type") not in exclusion_list)
            after = len(ds)
            print(f"  Filtered to classical: {before - after:,} colloquial removed -> {after:,} remaining.")

        all_chunks = []
        for row in ds:
            chunks = build_chunks(row)
            all_chunks.extend(chunks)
    else:
        # ── Streaming path (bounded memory, limit counts matching poems) ───
        dropped = {"empty": 0, "colloquial": 0, "whitelist": 0}
        matched_poems = 0
        all_chunks = []
        for row in ds:
            if not is_valid_row(row):
                dropped["empty"] += 1
                continue
            if classical_only and row.get("poem language type") in exclusion_list:
                dropped["colloquial"] += 1
                continue
            if poet_whitelist and row.get("poet name") not in poet_whitelist:
                dropped["whitelist"] += 1
                continue
            chunks = build_chunks(row)
            if quality_only:
                chunks = [
                    c for c in chunks
                    if c["is_orphan"] is False
                    and "..." not in c["text_index"]
                    and "\u2026" not in c["text_index"]
                ]
            all_chunks.extend(chunks)
            matched_poems += 1
            if limit and matched_poems >= limit:
                print(f"  Reached {limit} matching poems; stopping the scan early.")
                break
        print(
            "  Streaming scan: "
            f"kept {matched_poems:,} poems; "
            f"dropped empty={dropped['empty']:,}, "
            f"colloquial={dropped['colloquial']:,}, "
            f"off-whitelist={dropped['whitelist']:,}."
        )

    total_poems = matched_poems if streaming else len(ds)
    total_chunks = len(all_chunks)
    orphans = sum(1 for c in all_chunks if c["is_orphan"])
    print(f"\nDone.")
    print(f"  Poems processed : {total_poems:,}")
    print(f"  Total chunks    : {total_chunks:,}  (one per بيت)")
    print(f"  Orphan chunks   : {orphans:,}  ({orphans/total_chunks*100:.1f}% - single-hemistich verses)")

    return all_chunks


# ── Quick sanity-check entrypoint ─────────────────────────────────────────────

if __name__ == "__main__":
    from collections import Counter

    # Run on a small slice first so you can read the output
    chunks = run_pipeline(limit=200)

    print("\n-- Sample chunk (first verse of first poem) --")
    c = chunks[0]
    print(f"  display : {c['text_display']}")
    print(f"  index   : {c['text_index']}")
    print(f"  title   : {c['poem_title']}")
    print(f"  poet    : {c['poet_name']}")
    print(f"  era     : {c['poet_era']}")
    print(f"  meter   : {c['poem_meter']}")
    print(f"  theme   : {c['poem_theme']}")
    print(f"  lang    : {c['poem_language_type']}")
    print(f"  orphan  : {c['is_orphan']}")

    print("\n-- Language type distribution (top 10) --")
    lang_counts = Counter(c["poem_language_type"] for c in chunks)
    for lang, count in lang_counts.most_common(10):
        print(f"  {str(lang)!r:30s} {count:>6,}")

    print("\n-- Normalization spot-check --")
    test_cases = [
        "أَصبَحَ المُلكُ لِلَّذي فَطَرَ الخَلقَ",
        "إِنَّ اللَّهَ لا يَخفى عَلَيهِ شَيءٌ",
        "آمَنَ الرَّسولُ بِما أُنزِلَ إِلَيهِ",
    ]
    for t in test_cases:
        print(f"  original  : {t}")
        print(f"  normalized: {normalize_arabic(t)}")
        print()
