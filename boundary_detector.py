"""
Boundary detector abstraction layer.

Provides a common interface for different boundary detection models,
allowing easy swapping between implementations (e.g., Canine vs. Flair).
"""

from abc import ABC, abstractmethod
from typing import Optional
import torch
from pathlib import Path
import json


class BoundaryDetector(ABC):
    """
    Abstract base class for boundary detection models.
    
    All boundary detectors must implement the predict_boundaries method,
    which takes a list of line dictionaries and returns them annotated
    with 'predicted_nonbreaking_next_line' field.
    """
    
    @abstractmethod
    def predict_boundaries(
        self,
        lines: list[dict],
        threshold: float = 0.6,
        context_chars: int = 40,
    ) -> list[dict]:
        """
        Predict nonbreaking line boundaries.
        
        Args:
            lines: List of line dicts with at minimum: id, doc_id, source_sic
            threshold: Confidence threshold for nonbreaking prediction
            context_chars: Number of characters to use from each line end/start
            
        Returns:
            List of line dicts with 'predicted_nonbreaking_next_line' field added
        """
        pass
    
    @abstractmethod
    def get_default_threshold(self) -> float:
        """Return the default threshold for this detector."""
        pass


class CanineBoundaryDetector(BoundaryDetector):
    """
    Boundary detector using the custom Canine-based classifier.
    
    This is the original implementation from boundary_classifier.py.
    """
    
    def __init__(
        self,
        model_path: str,
        use_lexicon: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Canine boundary detector.
        
        Args:
            model_path: Directory containing best_model.pt and threshold.json,
                       or Hugging Face model identifier (e.g., mpilhlt/canine-salamanca-boundary-classifier)
            use_lexicon: Whether to use lexicon features
            device: Torch device to use (defaults to cuda/cpu)
        """
        from boundary_classifier.boundary_classifier import BoundaryClassifier
        from transformers import CanineTokenizer
        from data.data_utils import CorpusLexicon
        from huggingface_hub import hf_hub_download
        
        self.model_path = model_path
        self.model_dir = Path(model_path)
        self.use_lexicon = use_lexicon
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Load tokenizer
        print("Loading Canine tokenizer...")
        self.tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
        
        # Load model
        print("Loading Canine boundary classifier...")
        self.model = BoundaryClassifier(use_lexicon=use_lexicon)
        
        # Check if model_path is a local directory or HF identifier
        if self.model_dir.exists():
            # Local directory
            weights_path = self.model_dir / "best_model.pt"
            if not weights_path.exists():
                raise FileNotFoundError(f"Missing boundary weights: {weights_path}")
            
            self.model.load_state_dict(
                torch.load(weights_path, map_location=device)
            )
            print(f"Canine boundary detector loaded from local directory: {model_path}")
        else:
            # Hugging Face identifier
            print(f"Loading Canine boundary detector from Hugging Face: {model_path}")
            try:
                # Download model weights from Hugging Face
                weights_path = hf_hub_download(
                    repo_id=model_path,
                    filename="best_model.pt",
                    repo_type="model"
                )
                self.model.load_state_dict(
                    torch.load(weights_path, map_location=device)
                )
                print(f"Successfully loaded model weights from Hugging Face")
            except Exception as e:
                raise ValueError(
                    f"Failed to load model from Hugging Face: {e}\n"
                    f"Ensure the model identifier is correct and the model contains best_model.pt"
                )
        
        self.model.to(device)
        self.model.eval()
        
        # Load threshold
        if self.model_dir.exists():
            # Local directory
            threshold_path = self.model_dir / "threshold.json"
            if threshold_path.exists():
                self.default_threshold = json.loads(threshold_path.read_text())["threshold"]
            else:
                self.default_threshold = 0.6
                print(f"No threshold found; using default {self.default_threshold}")
        else:
            # Hugging Face identifier
            try:
                threshold_path = hf_hub_download(
                    repo_id=model_path,
                    filename="threshold.json",
                    repo_type="model"
                )
                self.default_threshold = json.loads(Path(threshold_path).read_text())["threshold"]
                print(f"Loaded threshold from Hugging Face: {self.default_threshold}")
            except Exception as e:
                print(f"Warning: Could not load threshold from Hugging Face: {e}")
                print(f"Using default threshold: 0.6")
                self.default_threshold = 0.6
        
        # Load lexicon if needed
        self.lexicon = None
        if use_lexicon:
            # Note: lexicon needs to be built from training data
            # This is handled externally via the lexicon_data_path parameter
            pass
    
    def predict_boundaries(
        self,
        lines: list[dict],
        threshold: float = 0.6,
        context_chars: int = 40,
    ) -> list[dict]:
        """Predict boundaries using Canine model."""
        from boundary_classifier.boundary_classifier import predict_boundaries
        from collections import defaultdict
        
        # Group lines by document
        by_doc: dict = defaultdict(list)
        for row in lines:
            by_doc[row["doc_id"]].append(row)
        
        predictions: dict = {}
        
        for doc_lines in by_doc.values():
            for i in range(len(doc_lines) - 1):
                row = doc_lines[i]
                next_row = doc_lines[i + 1]
                text = (
                    row["source_sic"][-context_chars:]
                    + "↵"
                    + next_row["source_sic"][:context_chars]
                )
                
                enc = self.tokenizer(
                    text,
                    max_length=128,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                lexicon_hit = None
                if self.lexicon is not None:
                    lexicon_hit = torch.tensor([[
                        float(self.lexicon.concatenation_is_known(
                            row["source_sic"], next_row["source_sic"]
                        ))
                    ]])
                
                with torch.no_grad():
                    out = self.model(
                        input_ids=enc["input_ids"].to(self.device),
                        attention_mask=enc["attention_mask"].to(self.device),
                        lexicon_hit=lexicon_hit,
                    )
                prob = torch.sigmoid(out["logits"]).item()
                if prob >= threshold:
                    predictions[row["id"]] = next_row["id"]
        
        result = []
        for row in lines:
            row = dict(row)
            row["predicted_nonbreaking_next_line"] = predictions.get(row["id"], "")
            result.append(row)
        
        return result
    
    def get_default_threshold(self) -> float:
        return self.default_threshold
    
    def set_lexicon(self, lexicon):
        """Set the lexicon for this detector."""
        self.lexicon = lexicon


class FlairBoundaryDetector(BoundaryDetector):
    """
    Boundary detector using Flair sequence tagger.
    
    Uses the mschonhardt/latin-contextual-lb-detector model from Hugging Face.
    """
    
    def __init__(
        self,
        model_name: str = "mschonhardt/latin-contextual-lb-detector",
        device: Optional[str] = None,
    ):
        """
        Initialize Flair boundary detector.
        
        Args:
            model_name: Hugging Face model name or path
            device: Device to use (e.g., 'cpu', 'cuda:0')
        """
        try:
            from flair.models import SequenceTagger
            from flair.data import Sentence
        except ImportError:
            raise ImportError(
                "Flair is required for FlairBoundaryDetector. "
                "Install with: pip install flair"
            )
        
        self.model_name = model_name
        self.Sentence = Sentence
        self.default_threshold = 0.5  # Default for Flair confidence
        
        print(f"Loading Flair model: {model_name} ...")
        self.tagger = SequenceTagger.load(model_name)
        
        if device:
            self.tagger.to(device)
        
        print("Flair boundary detector loaded successfully!")
    
    def predict_boundaries(
        self,
        lines: list[dict],
        threshold: float = 0.5,
        context_chars: int = 40,
    ) -> list[dict]:
        """
        Predict boundaries using Flair model.
        
        Note: context_chars is not used by Flair as it processes full context.
        """
        from collections import defaultdict
        
        # Group lines by document
        by_doc: dict = defaultdict(list)
        for row in lines:
            by_doc[row["doc_id"]].append(row)
        
        predictions: dict = {}
        
        for doc_lines in by_doc.values():
            for i in range(len(doc_lines) - 1):
                row = doc_lines[i]
                next_row = doc_lines[i + 1]
                
                # Create input in Flair format: line_end <lb/> line_start
                # Use context_chars to limit input size
                line_end = row["source_sic"][-context_chars:]
                line_start = next_row["source_sic"][:context_chars]
                text_input = f"{line_end} <lb/> {line_start}"
                
                # Tokenize by whitespace to preserve <lb/> as single token
                token_list = text_input.split()
                if not token_list:
                    continue
                
                # Create Flair Sentence
                sentence = self.Sentence(token_list)
                
                # Predict
                self.tagger.predict(sentence)
                
                # Extract prediction for <lb/> token
                for token in sentence:
                    if "<lb" in token.text:
                        tag = token.get_label().value
                        confidence = token.get_label().score
                        
                        # WB = Split Word (Join/Nonbreaking)
                        # NB = Separate Words (Space/Breaking)
                        if tag == "WB" and confidence >= threshold:
                            predictions[row["id"]] = next_row["id"]
                        break
        
        result = []
        for row in lines:
            row = dict(row)
            row["predicted_nonbreaking_next_line"] = predictions.get(row["id"], "")
            result.append(row)
        
        return result
    
    def get_default_threshold(self) -> float:
        return self.default_threshold


def create_boundary_detector(
    model_type: str,
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    use_lexicon: bool = False,
    device: Optional[torch.device] = None,
) -> BoundaryDetector:
    """
    Factory function to create a boundary detector.
    
    Args:
        model_type: Type of detector ('canine' or 'flair')
        model_path: Directory with best_model.pt (for canine) or HF model identifier
        model_name: Hugging Face model name (for flair, alias for model_path)
        use_lexicon: Whether to use lexicon (for canine)
        device: Torch device to use
        
    Returns:
        BoundaryDetector instance
    """
    model_type = model_type.lower()
    
    if model_type == "canine":
        # For canine, use model_path (or model_name as fallback)
        path = model_path or model_name
        if path is None:
            raise ValueError("model_path or model_name is required for canine detector")
        return CanineBoundaryDetector(
            model_path=path,
            use_lexicon=use_lexicon,
            device=device,
        )
    elif model_type == "flair":
        # For flair, use model_name (or model_path as fallback)
        name = model_name or model_path
        return FlairBoundaryDetector(
            model_name=name or "mschonhardt/latin-contextual-lb-detector",
            device=str(device) if device else None,
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported types: 'canine', 'flair'"
        )
