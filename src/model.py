from transformers import ViTForImageClassification
import torch
from torch import nn

class ViTClassifier(nn.Module):
    def __init__(self, num_classes = 1):
        super().__init__()
        self.classifier = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels = num_classes, ignore_mismatched_sizes = True)

    def forward(self, x):
        outputs = self.classifier(pixel_values = x)
        
        return outputs.logits