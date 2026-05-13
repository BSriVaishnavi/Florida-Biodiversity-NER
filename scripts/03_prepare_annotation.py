"""
03_prepare_annotation.py
Phase 4 — Select 300 most informative sentences for manual annotation.

Selects a diverse, balanced set of sentences that:
  - Have real user-written descriptions (not just metadata)
  - Cover diverse species
  - Contain entity-rich text
  - Are short enough to annotate efficiently

Exports in TWO formats:
  1. Label Studio JSON — import directly into Label Studio
  2. Simple CSV — annotate in Excel if Label Studio not available

Usage:
    python scripts/03_prepare_annotation.py
"""

import json
import csv
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT        = Path(__file__).resolve().parent.parent
PROCESSED   = ROOT / "data" / "processed"
ANNOTATIONS = ROOT / "data" / "annotations"
ANNOTATIONS.mkdir(parents=True, exist_ok=True)

random.seed(42)


def select_annotation_candidates(
    sentences_with_desc: list[dict],
    baseline_predictions: list[dict],
    n: int = 300
) -> list[dict]:
    """
    Select the most informative sentences for annotation.

    Strategy:
      - Must have real user-written description (not just metadata)
      - Prefer sentences with multiple entity types (richer annotation)
      - Balance across top species
      - Length between 30 and 200 chars (efficient to annotate)
      - Deduplicate by text
    """
    # Build set of sentences that had baseline predictions
    pred_by_sent = {p['sent_id']: p for p in baseline_predictions}

    # Filter candidates
    candidates = []
    seen_texts = set()

    for sent in sentences_with_desc:
        text = sent['text'].strip()

        # Length filter
        if len(text) < 30 or len(text) > 250:
            continue

        # Dedup
        if text.lower() in seen_texts:
            continue
        seen_texts.add(text.lower())

        # Score by entity richness
        pred = pred_by_sent.get(sent['sent_id'])
        n_entity_types = 0
        entities = []
        if pred:
            labels = set(e['label'] for e in pred['entities'])
            n_entity_types = len(labels)
            entities = pred['entities']

        candidates.append({
            **sent,
            "n_entity_types": n_entity_types,
            "entities":       entities,
            "score":          n_entity_types * 3 + len(text) / 100,
        })

    # Sort by score
    candidates.sort(key=lambda x: x['score'], reverse=True)

    # Balance across species (max 20 per species)
    species_count = defaultdict(int)
    selected = []
    max_per_species = 20

    for cand in candidates:
        species = cand.get('common_name', 'Unknown')
        if species_count[species] < max_per_species:
            selected.append(cand)
            species_count[species] += 1
        if len(selected) >= n:
            break

    # If not enough from entity-rich, fill with any description sentences
    if len(selected) < n:
        remaining = [c for c in candidates if c not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[:n - len(selected)])

    return selected[:n]


def export_label_studio(sentences: list[dict], path: Path):
    """
    Export in Label Studio JSON format.
    Import at: Settings → Labeling Interface → JSON

    Label config for Label Studio:
    <View>
      <Text name="text" value="$text"/>
      <Labels name="label" toName="text">
        <Label value="SPECIES" background="#1D9E75"/>
        <Label value="BEHAVIOR" background="#534AB7"/>
        <Label value="HABITAT" background="#D85A30"/>
      </Labels>
    </View>
    """
    tasks = []
    for i, sent in enumerate(sentences):
        task = {
            "id": i + 1,
            "data": {
                "text":            sent['text'],
                "sent_id":         sent['sent_id'],
                "observation_id":  sent['observation_id'],
                "common_name":     sent.get('common_name', ''),
                "scientific_name": sent.get('scientific_name', ''),
                "place_guess":     sent.get('place_guess', ''),
            },
            "predictions": [{
                "model_version": "rule_based_v1",
                "result": [
                    {
                        "id":            f"pred_{i}_{j}",
                        "type":          "labels",
                        "from_name":     "label",
                        "to_name":       "text",
                        "value": {
                            "start":  ent['start_char'],
                            "end":    ent['end_char'],
                            "text":   ent['text'],
                            "labels": [ent['label']],
                        }
                    }
                    for j, ent in enumerate(sent.get('entities', []))
                ]
            }] if sent.get('entities') else []
        }
        tasks.append(task)

    with open(path, 'w') as f:
        json.dump(tasks, f, indent=2)


def export_csv_for_annotation(sentences: list[dict], path: Path):
    """
    Export as CSV for manual annotation in Excel or Google Sheets.
    Columns: sent_id | text | SPECIES | BEHAVIOR | HABITAT | notes
    Pre-fills with baseline predictions for you to correct.
    """
    rows = []
    for sent in sentences:
        entities = sent.get('entities', [])
        species_pre  = '; '.join(set(e['text'] for e in entities if e['label']=='SPECIES'))
        behavior_pre = '; '.join(set(e['text'] for e in entities if e['label']=='BEHAVIOR'))
        habitat_pre  = '; '.join(set(e['text'] for e in entities if e['label']=='HABITAT'))

        rows.append({
            "sent_id":         sent['sent_id'],
            "observation_id":  sent['observation_id'],
            "common_name":     sent.get('common_name', ''),
            "text":            sent['text'],
            "SPECIES":         species_pre,
            "BEHAVIOR":        behavior_pre,
            "HABITAT":         habitat_pre,
            "notes":           "",
            "annotator":       "",
            "confidence":      "",
        })

    pd.DataFrame(rows).to_csv(path, index=False)


def export_conll(sentences: list[dict], path: Path):
    """
    Export in CoNLL-2003 BIO format for BERT fine-tuning.
    Each token on its own line: TOKEN LABEL
    Sentences separated by blank lines.
    """
    with open(path, 'w') as f:
        for sent in sentences:
            text     = sent['text']
            entities = sent.get('entities', [])
            tokens   = text.split()

            # Build character-level label map
            char_labels = ['O'] * len(text)
            for ent in entities:
                label = ent['label']
                start = ent['start_char']
                end   = ent['end_char']
                # Mark B- for first char, I- for rest
                char_labels[start] = f'B-{label}'
                for ci in range(start + 1, min(end, len(char_labels))):
                    char_labels[ci] = f'I-{label}'

            # Map tokens to labels
            pos = 0
            for token in tokens:
                token_start = text.find(token, pos)
                if token_start == -1:
                    f.write(f"{token} O\n")
                    pos += len(token) + 1
                    continue
                label = char_labels[token_start]
                f.write(f"{token} {label}\n")
                pos = token_start + len(token)

            f.write("\n")  # blank line between sentences


def main():
    print("=" * 60)
    print("Phase 4 — Preparing Annotation Set")
    print("=" * 60)

    # Load data
    with open(PROCESSED / "sentences_with_descriptions.json") as f:
        desc_sentences = json.load(f)
    with open(ROOT / "results" / "baseline_predictions.json") as f:
        predictions = json.load(f)

    print(f"  Sentences with descriptions : {len(desc_sentences):,}")
    print(f"  Baseline predictions        : {len(predictions):,}")

    # Select 300 best candidates
    print("\nSelecting 300 annotation candidates...")
    selected = select_annotation_candidates(desc_sentences, predictions, n=300)
    print(f"  Selected: {len(selected)} sentences")

    # Species distribution in selection
    from collections import Counter
    species_dist = Counter(s.get('common_name','Unknown') for s in selected)
    print("\n  Species distribution in annotation set:")
    for sp, cnt in species_dist.most_common(10):
        print(f"    {sp:<35} {cnt}")

    # Export Label Studio format
    ls_path = ANNOTATIONS / "label_studio_tasks.json"
    export_label_studio(selected, ls_path)
    print(f"\n  Saved Label Studio format : {ls_path}")

    # Export CSV format
    csv_path = ANNOTATIONS / "annotation_sheet.csv"
    export_csv_for_annotation(selected, csv_path)
    print(f"  Saved CSV format          : {csv_path}")

    # Export CoNLL format (pre-filled with baseline)
    conll_path = ANNOTATIONS / "baseline_conll.txt"
    export_conll(selected, conll_path)
    print(f"  Saved CoNLL format        : {conll_path}")

    # Save selected sentences
    with open(ANNOTATIONS / "selected_sentences.json", 'w') as f:
        json.dump(selected, f, indent=2)

    print(f"\n{'='*60}")
    print("ANNOTATION INSTRUCTIONS")
    print(f"{'='*60}")
    print("""
OPTION A — Label Studio (recommended, looks impressive):
  1. Go to https://labelstud.io and install locally
  2. Create new project
  3. Use this label config:

     <View>
       <Text name="text" value="$text"/>
       <Labels name="label" toName="text">
         <Label value="SPECIES"  background="#1D9E75"/>
         <Label value="BEHAVIOR" background="#534AB7"/>
         <Label value="HABITAT"  background="#D85A30"/>
       </Labels>
     </View>

  4. Import: data/annotations/label_studio_tasks.json
  5. Baseline predictions are pre-loaded — just correct them
  6. Export as JSON when done → save to data/annotations/annotated.json

OPTION B — CSV (faster, simpler):
  1. Open data/annotations/annotation_sheet.csv in Excel
  2. For each row, correct the SPECIES / BEHAVIOR / HABITAT columns
  3. Entities separated by semicolons e.g. "raccoon; deer"
  4. Save when done

ANNOTATION GUIDELINES:
  - SPECIES : any animal mentioned (common OR scientific name)
  - BEHAVIOR: what the animal is doing or how it was observed
  - HABITAT : where it was observed (place type, not specific address)
  - When unsure → leave blank (better than wrong label)
  - Overlapping entities → label the more specific one
""")

    print("✓ Annotation preparation complete.")
    return selected


if __name__ == "__main__":
    main()
