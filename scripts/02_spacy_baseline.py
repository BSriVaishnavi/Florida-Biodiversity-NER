"""
02_spacy_baseline.py — Phase 3: Rule-based NER Baseline
Entity types: SPECIES, BEHAVIOR, HABITAT
Run: python scripts/02_spacy_baseline.py
"""

import json, re, pandas as pd
from pathlib import Path
from collections import Counter

ROOT      = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS   = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SPECIES_VOCAB = [
    "eastern gray squirrel","gray squirrel","fox squirrel","squirrel",
    "common raccoon","raccoon",
    "white-tailed deer","white tailed deer","whitetail deer","key deer","deer",
    "marsh rabbit","eastern cottontail","cottontail","rabbit",
    "nine-banded armadillo","mexican long-nosed armadillo","armadillo",
    "florida manatee","west indian manatee","manatee","sea cow",
    "virginia opossum","opossum","possum",
    "common bottlenose dolphin","bottlenose dolphin","atlantic spotted dolphin","dolphin",
    "right whale","sperm whale","pygmy sperm whale","whale",
    "domestic cat","feral cat","cat",
    "florida panther","panther","bobcat",
    "north american river otter","river otter","otter",
    "wild boar","feral pig","wild pig","feral hog","boar",
    "florida black bear","black bear","bear",
    "coyote","red fox","gray fox","fox",
    "striped skunk","skunk","eastern mole","mole",
    "big brown bat","evening bat","bat",
    "white-footed mouse","florida mouse","mouse","mice",
    "eastern woodrat","woodrat","hispid cotton rat","cotton rat","rat",
    "round-tailed muskrat","muskrat","american mink","mink",
    "long-tailed weasel","weasel","southeastern pocket gopher","gopher",
    "west indian monk seal","monk seal","seal",
    # Scientific names
    "sciurus carolinensis","sciurus niger","procyon lotor",
    "odocoileus virginianus","odocoileus virginianus clavium",
    "sylvilagus palustris","sylvilagus floridanus","dasypus novemcinctus",
    "trichechus manatus","trichechus manatus latirostris",
    "didelphis virginiana","tursiops truncatus","stenella frontalis",
    "felis catus","puma concolor coryi","lynx rufus",
    "lontra canadensis","sus scrofa","ursus americanus floridanus",
    "canis latrans","vulpes vulpes","urocyon cinereoargenteus",
    "mephitis mephitis","scalopus aquaticus","eptesicus fuscus",
    "peromyscus floridanus","neotoma floridana","sigmodon hispidus",
    "neofiber alleni","mustela frenata","neovison vison","geomys pinetis",
]

BEHAVIOR_VOCAB = [
    "swimming","running","walking","jumping","climbing","flying","diving",
    "wading","crawling","trotting","galloping","foraging","grazing","browsing",
    "feeding","eating","drinking","hunting","stalking","chasing","preying",
    "scavenging","caching","rooting","digging","burrowing","playing","grooming",
    "fighting","mating","courting","nursing","resting","sleeping","basking",
    "sunbathing","calling","vocalizing","barking","howling","chirping",
    "marking territory","territorial","patrolling","crossing the road",
    "road crossing","crossing","traveling","migrating","dispersing",
    "nesting","denning","roosting","spotted","observed","seen","found",
    "encountered","photographed","documented","recorded","sighted",
]

HABITAT_VOCAB = [
    "pond","lake","river","creek","stream","canal","ditch","spring",
    "spring run","ocean","gulf","bay","lagoon","estuary","intracoastal",
    "waterway","wetland","swamp","marsh","bog","slough","cypress swamp",
    "mangrove","mangroves","sawgrass","everglades","forest","woodland",
    "hammock","oak hammock","hardwood hammock","pine flatwoods","flatwoods",
    "scrub","sandhill","prairie","savanna","grassland","field","meadow",
    "pasture","rangeland","beach","coastline","shore","shoreline",
    "seagrass bed","coral reef","reef","tidal flat","mudflat",
    "park","backyard","garden","yard","neighborhood","suburban","urban",
    "road","highway","trail","path","parking lot",
    "big cypress","ocala national forest","paynes prairie","myakka",
    "corkscrew swamp","florida keys","biscayne bay","tampa bay",
    "sarasota bay","st johns river","suwannee river","peace river",
    "lake okeechobee",
]


class RuleBasedNER:
    def __init__(self):
        self._compiled = {}
        for label, vocab in [("SPECIES", SPECIES_VOCAB),
                              ("BEHAVIOR", BEHAVIOR_VOCAB),
                              ("HABITAT", HABITAT_VOCAB)]:
            pats = sorted(set(vocab), key=len, reverse=True)
            self._compiled[label] = re.compile(
                r'\b(' + '|'.join(re.escape(p) for p in pats) + r')\b',
                re.IGNORECASE)

    def predict(self, text: str) -> list[dict]:
        raw = []
        for label, pat in self._compiled.items():
            for m in pat.finditer(text):
                raw.append({"text": m.group(), "label": label,
                             "start_char": m.start(), "end_char": m.end()})
        raw.sort(key=lambda x: (x['start_char'], -(x['end_char']-x['start_char'])))
        entities, last_end = [], -1
        for m in raw:
            if m['start_char'] >= last_end:
                entities.append(m)
                last_end = m['end_char']
        return entities

    def predict_batch(self, sentences: list[dict]) -> list[dict]:
        out = []
        for sent in sentences:
            ents = self.predict(sent['text'])
            if ents:
                out.append({**sent, "entities": ents, "n_entities": len(ents)})
        return out


def extract_triplets(predictions):
    triplets = []
    for pred in predictions:
        species  = [e['text'] for e in pred['entities'] if e['label']=='SPECIES']
        behaviors= [e['text'] for e in pred['entities'] if e['label']=='BEHAVIOR']
        habitats = [e['text'] for e in pred['entities'] if e['label']=='HABITAT']
        for sp in species:
            triplets.append({
                "observation_id":  pred['observation_id'],
                "species":         sp,
                "scientific_name": pred.get('scientific_name',''),
                "behaviors":       behaviors,
                "habitats":        habitats,
                "place_guess":     pred.get('place_guess',''),
                "latitude":        pred.get('latitude'),
                "longitude":       pred.get('longitude'),
                "source_text":     pred['text'],
                "n_behaviors":     len(behaviors),
                "n_habitats":      len(habitats),
            })
    return triplets


def main():
    print("="*60)
    print("Phase 3 — Rule-based NER Baseline")
    print("="*60)

    with open(PROCESSED/"sentences_all.json") as f:
        sentences = json.load(f)
    print(f"  Sentences to process: {len(sentences):,}")

    ner = RuleBasedNER()
    print(f"  Patterns loaded: SPECIES / BEHAVIOR / HABITAT")

    print("\nRunning NER...")
    predictions = ner.predict_batch(sentences)
    print(f"  Sentences with entities: {len(predictions):,}")

    all_ents   = [e for p in predictions for e in p['entities']]
    sp_ents    = [e for e in all_ents if e['label']=='SPECIES']
    beh_ents   = [e for e in all_ents if e['label']=='BEHAVIOR']
    hab_ents   = [e for e in all_ents if e['label']=='HABITAT']

    print(f"\n  SPECIES  : {len(sp_ents):,}")
    print(f"  BEHAVIOR : {len(beh_ents):,}")
    print(f"  HABITAT  : {len(hab_ents):,}")
    print(f"  TOTAL    : {len(all_ents):,}")

    print("\n  Top 10 SPECIES:")
    for sp, cnt in Counter(e['text'].lower() for e in sp_ents).most_common(10):
        print(f"    {sp:<35} {cnt}")

    print("\n  Top 10 BEHAVIORS:")
    for bh, cnt in Counter(e['text'].lower() for e in beh_ents).most_common(10):
        print(f"    {bh:<35} {cnt}")

    print("\n  Top 10 HABITATS:")
    for hb, cnt in Counter(e['text'].lower() for e in hab_ents).most_common(10):
        print(f"    {hb:<35} {cnt}")

    triplets = extract_triplets(predictions)
    print(f"\n  Structured triplets: {len(triplets):,}")

    with open(RESULTS/"baseline_predictions.json","w") as f:
        json.dump(predictions, f, indent=2)
    with open(RESULTS/"structured_triplets.json","w") as f:
        json.dump(triplets, f, indent=2)

    stats = {
        "total_sentences_with_entities": len(predictions),
        "total_entities": len(all_ents),
        "species_count": len(sp_ents),
        "behavior_count": len(beh_ents),
        "habitat_count": len(hab_ents),
        "total_triplets": len(triplets),
        "top_species":   Counter(e['text'].lower() for e in sp_ents).most_common(15),
        "top_behaviors": Counter(e['text'].lower() for e in beh_ents).most_common(15),
        "top_habitats":  Counter(e['text'].lower() for e in hab_ents).most_common(15),
    }
    with open(RESULTS/"baseline_stats.json","w") as f:
        json.dump(stats, f, indent=2)

    pd.DataFrame(triplets).to_csv(RESULTS/"triplets.csv", index=False)

    print("\nSample predictions:")
    for pred in predictions[:5]:
        print(f"\n  Text: {pred['text'][:100]}")
        for e in pred['entities']:
            print(f"    [{e['label']:8s}] '{e['text']}'")

    print(f"\n  Saved: results/baseline_predictions.json")
    print(f"  Saved: results/structured_triplets.json")
    print(f"  Saved: results/baseline_stats.json")
    print(f"  Saved: results/triplets.csv")
    print("\n✓ Baseline NER complete.")
    return predictions, triplets, ner


if __name__ == "__main__":
    main()
