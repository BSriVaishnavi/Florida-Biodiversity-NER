"""
run_pipeline.py
Master script — runs the full pipeline in order.

Usage:
    python run_pipeline.py                    # full pipeline
    python run_pipeline.py --use-baseline     # skip annotation, use baseline for BERT
    python run_pipeline.py --skip-bert        # skip BERT training
"""

import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_step(script: str, args: list = []):
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script)] + args,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\nERROR in {script}. Check output above.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-baseline', action='store_true',
                        help='Use baseline predictions for BERT training')
    parser.add_argument('--skip-bert', action='store_true',
                        help='Skip BERT fine-tuning')
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════╗
║     Florida Biodiversity NER — Full Pipeline             ║
║     iNaturalist Mammal Observations 2023-2024            ║
╚══════════════════════════════════════════════════════════╝
    """)

    # Phase 1 & 2: Preprocessing
    run_step("01_preprocess.py")

    # Phase 3: Baseline NER
    run_step("02_spacy_baseline.py")

    # Phase 4: Annotation preparation
    run_step("03_prepare_annotation.py")

    # Phase 5: BERT fine-tuning
    if not args.skip_bert:
        bert_args = ["--use-baseline"] if args.use_baseline else []
        run_step("04_bert_finetune.py", bert_args)
    else:
        print("\nSkipping BERT fine-tuning (--skip-bert)")

    # Phase 6: Evaluation
    run_step("05_evaluate.py")

    print(f"""
╔══════════════════════════════════════════════════════════╗
║     Pipeline Complete!                                   ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Results saved to: results/                              ║
║    baseline_predictions.json                             ║
║    structured_triplets.json                              ║
║    triplets.csv                                          ║
║    evaluation_report.md                                  ║
║    baseline_stats.json                                   ║
║                                                          ║
║  Annotations ready: data/annotations/                    ║
║    annotation_sheet.csv   ← open in Excel                ║
║    label_studio_tasks.json ← import to Label Studio      ║
║                                                          ║
║  To launch the query interface:                          ║
║    streamlit run app/app.py                              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
