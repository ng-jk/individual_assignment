import json

from PIL import Image

import predict
from cardioexplain.vision_model import build_model, save_checkpoint


def test_cli_accepts_image_and_writes_json_and_gradcam(tmp_path, capsys):
    image_path = tmp_path / "sample.png"
    checkpoint_path = tmp_path / "model.pt"
    output_dir = tmp_path / "predictions"
    Image.new("L", (256, 256), color=90).save(image_path)
    model = build_model(pretrained=False, train_backbone=True)
    save_checkpoint(checkpoint_path, model, 0.5, {"f1": 0.0}, [], {"test": {"images": 1}})

    exit_code = predict.main([
        "--image", str(image_path),
        "--checkpoint", str(checkpoint_path),
        "--output-dir", str(output_dir),
        "--device", "cpu",
    ])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["classification"] in {"above_threshold", "below_threshold"}
    assert len(list(output_dir.glob("sample_*_prediction.json"))) == 1
    assert len(list(output_dir.glob("sample_*_gradcam.png"))) == 1
