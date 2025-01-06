#!/usr/bin/env python3
import torch
from transformers import AutoModel, AutoTokenizer
import os

def convert_to_onnx():
    # Create testdata directory if it doesn't exist
    os.makedirs("testdata", exist_ok=True)

    # Load model and tokenizer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save tokenizer
    tokenizer.save_pretrained("testdata")

    # Create dummy input with max sequence length
    max_length = 512  # Maximum sequence length for this model
    text = "This is a test input for ONNX conversion"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)

    # Set model to evaluation mode
    model.eval()

    # Export to ONNX with dynamic axes for both batch and sequence length
    torch.onnx.export(
        model,
        tuple(inputs.values()),
        "testdata/model.onnx",
        input_names=list(inputs.keys()),
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'token_type_ids': {0: 'batch_size', 1: 'sequence'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence'},
            'pooler_output': {0: 'batch_size'}
        },
        opset_version=14,
        do_constant_folding=True
    )

if __name__ == "__main__":
    convert_to_onnx() 