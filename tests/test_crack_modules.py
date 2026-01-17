# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.nn.tasks import SegmentationModel


def test_crack_model_forward():
    try:
        from torchvision.ops import deform_conv2d  # noqa: F401
    except Exception:
        pytest.skip("torchvision.ops.deform_conv2d not available for DSConv")
    model = SegmentationModel("ultralytics/cfg/models/v8/yolov8n-seg-crack.yaml")
    model.eval()
    imgs = torch.zeros(1, 3, 640, 640)
    priors = torch.zeros(1, 2, 640, 640)
    with torch.no_grad():
        preds = model(imgs, priors=priors)
    assert isinstance(preds, tuple)
    assert preds[0].shape[0] == 1
