# Pretrained Model Loader for Image Classification

A Python script implementing the function `get_pretrained_model()` for loading and modifying **pretrained deep learning models**. It allows easy fine-tuning of popular architectures for classification tasks.

## Features

- Supports multiple **pretrained models**:
  - `alexnet`
  - `vgg11`
  - `googlenet`
  - `resnet18`
- **Flexible classification head modification**:  
  - Replaces the final fully connected layer (`fc` or `classifier`) to match the specified `num_classes`.
- **Option to load pretrained weights** (`pretrained=True` by default).

## Function

The script defines:

```python
def get_pretrained_model(model_name: str, num_classes: int, pretrained: bool=True):
