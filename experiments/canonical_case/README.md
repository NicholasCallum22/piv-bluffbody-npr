Purpose
-------
Baseline dataset for pipeline validation and uncertainty quantification.

How to run locally
------------------
1. Generate synthetic data:
   python experiments/canonical_case/generate_synthetic.py --metadata experiments/canonical_case/metadata.yaml --out data/canonical

2. Run pipeline (example):
   python src/pipeline/run_pipeline.py --config experiments/canonical_case/metadata.yaml --input data/canonical

Notes
-----
- The generator is deterministic when `random_seed` is set in the metadata.
- CI will run a minimal check that the metadata loads and the generator produces expected files.
