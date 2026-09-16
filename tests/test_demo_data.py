from cardioexplain.demo_data import create_synthetic_chexpert
from cardioexplain.imaging_data import load_chexpert_metadata, patient_level_split


def test_generated_demo_data_is_loadable_and_patient_separated(tmp_path):
    dataset_dir = create_synthetic_chexpert(tmp_path / "demo", patients=20)
    frame = load_chexpert_metadata(dataset_dir)
    splits = patient_level_split(frame)

    assert len(frame) == 20
    assert set(frame["target"]) == {0.0, 1.0}
    patient_sets = {name: set(part["patient_id"]) for name, part in splits.items()}
    assert patient_sets["train"].isdisjoint(patient_sets["validation"])
    assert patient_sets["train"].isdisjoint(patient_sets["test"])
    assert patient_sets["validation"].isdisjoint(patient_sets["test"])
