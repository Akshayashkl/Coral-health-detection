

import torch
import torchvision.models as models
import torch.nn as nn

torch_model_path = "C:\\newprograms\\All_Projects\\Corals_new\\Model\\all_models\\models\\efficientnet_2class.pth"
onnx_model_path = "efficientnet.onnx"

# Load model
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(torch_model_path, map_location='cpu'))
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy_input, onnx_model_path,
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    opset_version=11
)

print(f"✅ Exported ONNX model to {onnx_model_path}")
