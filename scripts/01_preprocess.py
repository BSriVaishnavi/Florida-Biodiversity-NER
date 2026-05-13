"""
01_preprocess.py
Phase 1 & 2 — Data loading, exploration, and preprocessing.

Loads raw iNaturalist observations, cleans text, tokenizes sentences,
and outputs a clean JSON corpus ready for NER.

Usage:
    python scripts/01_preprocess.py
"""

import os
import re
import json
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
RAW_CSV    = ROOT / "data" / "raw" / "observations.csv"
PROCESSED  = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean a raw iNaturalist description.
    Removes URLs, escape sequences, excessive whitespace,
    template boilerplate like 'Description:', 'Reference(s):'.
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove escape sequences
    text = text.replace('\\r\\n', ' ').replace('\\n', ' ').replace('\\r', ' ')
    # Remove boilerplate labels
    for label in ['Description:', 'Habitat:', 'Reference(s):', 'Behavior:',
                  'Notes:', 'Location:', 'Observer notes:']:
        text = text.replace(label, '')
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def build_observation_text(row: pd.Series) -> str:
    """
    Build a rich text string from all available fields.
    Even if description is empty, we construct meaningful text
    from species name, common name, and place.
    """
    parts = []

    # Add description if available
    desc = clean_text(str(row.get('description', '')))
    if len(desc) > 20:
        parts.append(desc)

    # Always add a structured sentence from metadata
    common  = str(row.get('common_name', '')).strip()
    species = str(row.get('scientific_name', '')).strip()
    place   = str(row.get('place_guess', '')).strip()

    if common and common != 'nan':
        if place and place != 'nan':
            parts.append(f"{common} ({species}) observed at {place}.")
        else:
            parts.append(f"{common} ({species}) observed in Florida.")

    return ' '.join(parts).strip()


def sentence_tokenize(text: str) -> list[str]:
    """
    Simple sentence tokenizer.
    Splits on '.', '!', '?' followed by whitespace + capital letter.
    Filters out sentences that are too short to be useful.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    # Clean and filter
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences


# ── Main processing ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 1 — Loading raw data")
    print("=" * 60)

    df = pd.read_csv(RAW_CSV)
    print(f"  Total observations    : {len(df):,}")
    print(f"  Columns               : {df.columns.tolist()}")
    print(f"  Has description       : {df['description'].notna().sum():,} "
          f"({df['description'].notna().sum()/len(df)*100:.1f}%)")
    print(f"  No description        : {df['description'].isna().sum():,} "
          f"({df['description'].isna().sum()/len(df)*100:.1f}%)")

    print("\nTop 15 species in dataset:")
    top_species = df['common_name'].value_counts().head(15)
    for species, count in top_species.items():
        print(f"  {species:<40} {count:>5}")

    print("\n" + "=" * 60)
    print("Phase 2 — Preprocessing & cleaning")
    print("=" * 60)

    records = []
    sentence_records = []

    df = df[pd.to_numeric(df["id"], errors="coerce").notna()]
    for _, row in df.iterrows():
        obs_text = build_observation_text(row)
        if not obs_text:
            continue

        sentences = sentence_tokenize(obs_text)
        if not sentences:
            continue

        record = {
            "id":              int(row['id']),
            "quality_grade":   str(row.get('quality_grade', '')),
            "common_name":     str(row.get('common_name', '')),
            "scientific_name": str(row.get('scientific_name', '')),
            "place_guess":     str(row.get('place_guess', '')),
            "latitude":        float(row['latitude']) if pd.notna(row.get('latitude')) else None,
            "longitude":       float(row['longitude']) if pd.notna(row.get('longitude')) else None,
            "full_text":       obs_text,
            "sentences":       sentences,
            "has_description": pd.notna(row.get('description')) and len(str(row.get('description', ''))) > 20,
        }
        records.append(record)

        # Also build flat sentence list for NER
        for i, sent in enumerate(sentences):
            sentence_records.append({
                "sent_id":         f"{int(row['id'])}_{i}",
                "observation_id":  int(row['id']),
                "text":            sent,
                "common_name":     str(row.get('common_name', '')),
                "scientific_name": str(row.get('scientific_name', '')),
                "place_guess":     str(row.get('place_guess', '')),
                "latitude":        float(row['latitude']) if pd.notna(row.get('latitude')) else None,
                "longitude":       float(row['longitude']) if pd.notna(row.get('longitude')) else None,
                "has_description": pd.notna(row.get('description')) and len(str(row.get('description', ''))) > 20,
            })

    print(f"  Processed observations : {len(records):,}")
    print(f"  Total sentences        : {len(sentence_records):,}")

    # Filter to sentences with real descriptions for annotation
    desc_sentences = [s for s in sentence_records if s['has_description']]
    print(f"  Sentences w/ real desc : {len(desc_sentences):,}")

    # Save outputs
    obs_path  = PROCESSED / "observations_clean.json"
    sent_path = PROCESSED / "sentences_all.json"
    desc_path = PROCESSED / "sentences_with_descriptions.json"

    with open(obs_path,  'w') as f: json.dump(records,          f, indent=2)
    with open(sent_path, 'w') as f: json.dump(sentence_records, f, indent=2)
    with open(desc_path, 'w') as f: json.dump(desc_sentences,   f, indent=2)

    print(f"\n  Saved: {obs_path}")
    print(f"  Saved: {sent_path}")
    print(f"  Saved: {desc_path}")

    # Species distribution
    species_df = pd.DataFrame(records)
    species_dist = species_df['common_name'].value_counts()
    species_dist.to_csv(PROCESSED / "species_distribution.csv")
    print(f"  Saved: {PROCESSED / 'species_distribution.csv'}")

    print("\n✓ Preprocessing complete.")
    return records, sentence_records


if __name__ == "__main__":
    main()
