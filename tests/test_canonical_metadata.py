import yaml
from pathlib import Path
def test_metadata_has_required_fields():
    p = Path("experiments/canonical_case/metadata.yaml")
    assert p.exists(), "metadata.yaml missing"
    m = yaml.safe_load(p.read_text())
    assert "acquisition" in m
    assert "processing" in m
    assert "random_seed" in m
    assert isinstance(m["random_seed"], int)
