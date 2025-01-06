#!/usr/bin/env python3
import os
import sys
import torch
import json
import shutil
import typer
import logging
from typing import Dict, Optional, List
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
import psutil
import numpy as np
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import onnx
import onnxruntime as ort
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging with rich
console = Console()
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    path: str
    dim: int
    max_seq_length: int
    opset: int
    description: str
    normalized: bool
    model_type: str
    supports_metal: bool = True
    requires_tokenizer: bool = True
    quantization_supported: bool = True

# Expanded model dictionary with more metadata and models
SUPPORTED_MODELS = {
    "all-mpnet-base-v2": ModelConfig(
        path="sentence-transformers/all-mpnet-base-v2",
        dim=768,
        max_seq_length=512,
        opset=12,
        description="General purpose embeddings with high quality",
        normalized=True,
        model_type="sentence-transformer"
    ),
    "all-MiniLM-L6-v2": ModelConfig(
        path="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
        max_seq_length=512,
        opset=14,
        description="Fast, efficient embeddings for production",
        normalized=True,
        model_type="sentence-transformer"
    ),
    "bge-large-en-v1.5": ModelConfig(
        path="BAAI/bge-large-en-v1.5",
        dim=1024,
        max_seq_length=512,
        opset=14,
        description="Latest BGE model with strong performance",
        normalized=True,
        model_type="sentence-transformer",
        supports_metal=False  # Disable optimization for this model
    ),
    "e5-large-v2": ModelConfig(
        path="intfloat/e5-large-v2",
        dim=1024,
        max_seq_length=512,
        opset=14,
        description="State-of-the-art embeddings",
        normalized=True,
        model_type="sentence-transformer",
        supports_metal=False  # Disable optimization for this model
    ),
    # New multilingual model
    "paraphrase-multilingual-MiniLM-L12-v2": ModelConfig(
        path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        dim=384,
        max_seq_length=512,
        opset=14,
        description="High-quality multilingual embeddings (50+ languages)",
        normalized=True,
        model_type="sentence-transformer"
    ),
    
    # New specialized models
    "all-roberta-large-v1": ModelConfig(
        path="sentence-transformers/all-roberta-large-v1",
        dim=1024,
        max_seq_length=512,
        opset=14,
        description="Maximum accuracy model for high-quality embeddings",
        normalized=True,
        model_type="sentence-transformer"
    ),
    "multi-qa-MiniLM-L6-cos-v1": ModelConfig(
        path="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        dim=384,
        max_seq_length=512,
        opset=14,
        description="Optimized for question-answering and search",
        normalized=True,
        model_type="sentence-transformer"
    )
}

class ONNXConverter:
    def __init__(self, model_name: str, output_dir: str, config: dict):
        self.model_name = model_name
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(SUPPORTED_MODELS.keys())}")
        
        self.model_info = SUPPORTED_MODELS[model_name]
        self.output_dir = Path(output_dir)
        self.config = config
        self.device = "cuda" if config.get("use_cuda", False) and torch.cuda.is_available() else "cpu"
        
        # Setup directories
        self.model_dir = self.output_dir / model_name
        self.tokenizer_dir = self.model_dir / "tokenizer"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_dir.mkdir(parents=True, exist_ok=True)

    def convert(self) -> bool:
        """Main conversion process."""
        try:
            with console.status(f"Converting {self.model_name}...") as status:
                # Track memory usage
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Load model and tokenizer
                status.update(f"Loading {self.model_name}")
                model, tokenizer = self._load_model_and_tokenizer()
                
                # Save tokenizer files
                status.update("Saving tokenizer")
                self._save_tokenizer(tokenizer)
                
                # Convert to ONNX
                status.update("Converting to ONNX")
                self._convert_to_onnx(model, tokenizer)
                
                # Optimize if requested
                if self.config.get("optimize", False):
                    status.update("Optimizing model")
                    self._optimize_model()
                
                # Quantize if requested
                if self.config.get("quantize", False):
                    status.update("Quantizing model")
                    self._quantize_model()
                
                # Verify the model
                status.update("Verifying model")
                self._verify_model()
                
                # Clean up
                del model
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Log memory usage
                final_memory = psutil.Process().memory_info().rss / 1024 / 1024
                logger.info(f"Memory usage: {final_memory - initial_memory:.2f}MB")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to convert {self.model_name}: {str(e)}")
            return False

    def _load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load the model and tokenizer from HuggingFace."""
        try:
            # Get token from environment
            token = os.getenv('HUGGINGFACE_API_KEY')
            if not token:
                raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
                
            # Load model and tokenizer with auth token
            model = AutoModel.from_pretrained(
                self.model_info.path,
                trust_remote_code=True,
                use_auth_token=token,
                cache_dir="./.cache"  # Keep models in project directory
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_info.path,
                trust_remote_code=True,
                use_auth_token=token,
                cache_dir="./.cache"  # Keep models in project directory
            )
            
            model = model.to(self.device)
            model.eval()
            
            return model, tokenizer
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model and tokenizer: {str(e)}")

    def _save_tokenizer(self, tokenizer: PreTrainedTokenizer) -> None:
        """Save tokenizer files and configuration."""
        try:
            # Save tokenizer files
            tokenizer.save_pretrained(self.tokenizer_dir)
            
            # Create enhanced tokenizer config
            tokenizer_config = {
                "do_lower_case": getattr(tokenizer, "do_lower_case", False),
                "vocab_size": tokenizer.vocab_size,
                "pad_token": tokenizer.pad_token,
                "pad_token_id": tokenizer.pad_token_id,
                "max_model_input_sizes": tokenizer.model_max_length,
                "model_max_length": self.model_info.max_seq_length,
                "embedding_dim": self.model_info.dim,
                "normalized_embeddings": self.model_info.normalized,
                "special_tokens": {
                    "pad_token": tokenizer.pad_token,
                    "unk_token": tokenizer.unk_token,
                    "cls_token": tokenizer.cls_token if hasattr(tokenizer, "cls_token") else None,
                    "sep_token": tokenizer.sep_token if hasattr(tokenizer, "sep_token") else None,
                    "mask_token": tokenizer.mask_token if hasattr(tokenizer, "mask_token") else None
                },
                "model_type": self.model_info.model_type,
                "supports_metal": self.model_info.supports_metal,
                "quantization_supported": self.model_info.quantization_supported
            }
            
            # Save enhanced config
            config_path = self.tokenizer_dir / "tokenizer_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            raise RuntimeError(f"Failed to save tokenizer: {str(e)}")

    def _convert_to_onnx(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        """Convert the model to ONNX format."""
        try:
            # Prepare dummy input
            text = "This is a test input for ONNX conversion"
            encoded = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=self.model_info.max_seq_length,
                truncation=True
            )
            
            # Move inputs to correct device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Set output path
            output_path = self.model_dir / f"{self.model_name}.onnx"
            
            # Define dynamic axes
            dynamic_axes = {
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'token_type_ids': {0: 'batch_size'} if 'token_type_ids' in encoded else None,
                'last_hidden_state': {0: 'batch_size'},
                'pooler_output': {0: 'batch_size'}
            }
            dynamic_axes = {k: v for k, v in dynamic_axes.items() if v is not None}
            
            # Export to ONNX
            torch.onnx.export(
                model,
                tuple(encoded.values()),
                output_path,
                input_names=list(encoded.keys()),
                output_names=["last_hidden_state", "pooler_output"],
                dynamic_axes=dynamic_axes,
                opset_version=self.model_info.opset,
                do_constant_folding=True,
                verbose=False
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert model to ONNX: {str(e)}")

    def _optimize_model(self) -> None:
        """Optimize the ONNX model."""
        try:
            model_path = self.model_dir / f"{self.model_name}.onnx"
            optimized_path = self.model_dir / f"{self.model_name}_optimized.onnx"
            
            # Load the model
            model = onnx.load(model_path)
            
            # Basic optimizations
            from onnxruntime.transformers import optimizer
            opt_model = optimizer.optimize_model(
                str(model_path),
                model_type='bert',
                num_heads=12,  # Adjust based on model
                hidden_size=self.model_info.dim
            )
            
            # Save optimized model
            opt_model.save_model_to_file(str(optimized_path))
            
        except Exception as e:
            raise RuntimeError(f"Failed to optimize model: {str(e)}")

    def _quantize_model(self) -> None:
        """Quantize the ONNX model for reduced size and faster inference."""
        try:
            from onnxruntime.quantization import quantize_dynamic
            
            model_path = self.model_dir / f"{self.model_name}.onnx"
            quantized_path = self.model_dir / f"{self.model_name}_quantized.onnx"
            
            quantize_dynamic(
                str(model_path),
                str(quantized_path),
                weight_type=onnx.TensorProto.INT8
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to quantize model: {str(e)}")

    def _verify_model(self) -> None:
        """Verify the exported ONNX model."""
        try:
            model_path = self.model_dir / f"{self.model_name}.onnx"
            
            # Load and check ONNX model
            onnx_model = onnx.load(model_path)
            onnx.checker.check_model(onnx_model)
            
            # Test with ONNX Runtime
            session = ort.InferenceSession(str(model_path))
            
            # Save model metadata
            metadata = {
                "framework": "pytorch",
                "format": "onnx",
                "model_name": self.model_name,
                "original_model": self.model_info.path,
                "opset_version": self.model_info.opset,
                "input_names": [input.name for input in session.get_inputs()],
                "input_shapes": {input.name: input.shape for input in session.get_inputs()},
                "output_names": [output.name for output in session.get_outputs()],
                "output_shapes": {output.name: output.shape for output in session.get_outputs()},
                "description": self.model_info.description,
                "embedding_dimension": self.model_info.dim,
                "max_sequence_length": self.model_info.max_seq_length,
                "normalized_embeddings": self.model_info.normalized,
                "supports_metal": self.model_info.supports_metal,
                "quantization_supported": self.model_info.quantization_supported
            }
            
            metadata_path = self.model_dir / "model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            raise RuntimeError(f"Model verification failed: {str(e)}")

def convert_models_parallel(models: list, output_dir: str, config: dict) -> None:
    """Convert multiple models in parallel."""
    with ProcessPoolExecutor(max_workers=config['max_workers']) as executor:
        # Start conversion tasks
        future_to_model = {}
        for model in models:
            # Create converter instance
            converter = ONNXConverter(model, output_dir, config)
            # Submit the convert method of the instance
            future = executor.submit(converter.convert)
            future_to_model[future] = model
        
        # Track progress
        total_models = len(models)
        completed = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Converting models...", total=total_models)
            
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    success = future.result()
                    if success:
                        console.print(f"✓ Converted {model}")
                    else:
                        console.print(f"⚠ Failed to convert {model}")
                except Exception as e:
                    console.print(f"[red]Error converting {model}: {str(e)}")
                
                completed += 1
                progress.update(task, completed=completed)

def main(
    models: Optional[List[str]] = typer.Argument(None, help="Models to convert"),
    output_dir: str = typer.Option("models", help="Output directory"),
    use_cuda: bool = typer.Option(False, help="Use CUDA for conversion"),
    optimize: bool = typer.Option(False, help="Optimize models after conversion"),
    quantize: bool = typer.Option(False, help="Quantize models after conversion"),
    max_workers: int = typer.Option(2, help="Maximum number of parallel conversions"),
    list_models: bool = typer.Option(False, help="List available models"),
    validate: bool = typer.Option(True, help="Validate models after conversion")
):
    """Convert transformer models to ONNX format with various optimizations."""
    
    try:
        if list_models:
            table = Table(title="Available Models")
            table.add_column("Name")
            table.add_column("Description")
            table.add_column("Dimensions")
            table.add_column("Sequence Length")
            table.add_column("Metal Support")
            
            for name, info in SUPPORTED_MODELS.items():
                table.add_row(
                    name,
                    info.description,
                    str(info.dim),
                    str(info.max_seq_length),
                    "✓" if info.supports_metal else "✗"
                )
            
            console.print(table)
            return

        if not models:
            models = list(SUPPORTED_MODELS.keys())

        config = {
            "use_cuda": use_cuda,
            "optimize": optimize,
            "quantize": quantize,
            "max_workers": max_workers,
            "validate": validate
        }

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Convert models in parallel
        convert_models_parallel(models, output_dir, config)

        console.print("\n✨ Conversion complete!")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    typer.run(main)