from torchinfo import summary
from my_model import NuLite

# Create model
model = NuLite(num_classes=6, use_attention_gate=False)

# Print summary
summary(
    model,
    input_size=(1, 3, 256, 256),   # batch_size, channels, height, width
    col_names=["input_size", "output_size", "num_params"],
    depth=3
)
