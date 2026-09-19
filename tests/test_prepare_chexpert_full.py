import io
import zipfile
from argparse import Namespace

import pandas as pd
from PIL import Image

from prepare_chexpert_full import prepare


def _jpeg_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(stream, format="JPEG")
    return stream.getvalue()


def test_prepare_full_release_batches(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    source.mkdir()
    rows = []
    image_bytes = _jpeg_bytes()

    with zipfile.ZipFile(source / "CheXpert-v1.0 batch 1 (validate & csv).zip", "w") as bundle:
        for batch, patient in [(2, "patient00001"), (3, "patient20000"), (4, "patient43018")]:
            rows.append({
                "Path": f"CheXpert-v1.0/train/{patient}/study1/view1_frontal.jpg",
                "Frontal/Lateral": "Frontal",
                "Cardiomegaly": float(batch % 2),
            })
        rows.append({
            "Path": "CheXpert-v1.0/train/patient00002/study1/view1_lateral.jpg",
            "Frontal/Lateral": "Lateral",
            "Cardiomegaly": 1.0,
        })
        csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        bundle.writestr("CheXpert-v1.0 batch 1 (validate & csv)/train.csv", csv_bytes)

    extracted = source / "CheXpert-v1.0 batch 2 (train 1)"
    extracted_image = extracted / "patient00001/study1/view1_frontal.jpg"
    extracted_image.parent.mkdir(parents=True)
    extracted_image.write_bytes(image_bytes)
    unused_image = extracted / "patient00002/study1/view1_lateral.jpg"
    unused_image.parent.mkdir(parents=True)
    unused_image.write_bytes(image_bytes)

    for batch, label, patient in [(3, "train 2", "patient20000"),
                                  (4, "train 3", "patient43018")]:
        archive = source / f"CheXpert-v1.0 batch {batch} ({label}).zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(
                f"CheXpert-v1.0 batch {batch} ({label})/{patient}/study1/view1_frontal.jpg",
                image_bytes,
            )

    manifest = prepare(Namespace(
        source_dir=source, output_dir=output, size=224, quality=90, workers=2,
        batches=[2, 3, 4], overwrite=False, allow_incomplete=False,
    ))

    assert manifest["prepared_rows"] == 3
    assert manifest["missing_rows"] == 0
    assert manifest["metadata_rows"] == 4
    assert manifest["eligible_rows"] == 3
    assert sum(batch["images"] for batch in manifest["batches"]) == 3
    frame = pd.read_csv(output / "train.csv")
    assert frame["Path"].tolist() == [
        "train/patient00001/study1/view1_frontal.jpg",
        "train/patient20000/study1/view1_frontal.jpg",
        "train/patient43018/study1/view1_frontal.jpg",
    ]
    with Image.open(output / frame.iloc[0]["Path"]) as image:
        assert image.size == (224, 224)
