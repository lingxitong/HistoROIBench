

import os
import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from model_utils.model_factory import encoder_factory
from .common_utils import save_results_as_txt


def load_class_names_from_txt(txt_path: str) -> List[str]:
    """
    Load class names from txt file
    
    Args:
        txt_path: Path to txt file
        
    Returns:
        List of class names
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    class_names = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            # Assume format is "id:class_name" or directly class name
            if ':' in line:
                class_name = line.split(':', 1)[1].strip()
            else:
                class_name = line
            class_names.append(class_name)
    
    return class_names


def zero_shot_classify_with_features(model, 
                                    image_features: torch.Tensor,
                                    text_prompts: List[str],
                                    device: str) -> Dict:
    """
    Zero-shot classification - directly use extracted image features
    
    Args:
        model: Model instance (with encode_text method and logit_scale attribute)
        image_features: Extracted image features [num_images, dim]
        text_prompts: List of complete text prompts, each element corresponds to a class
        device: Device
        
    Returns:
        Classification result dictionary
    """
    # Ensure image features are on the correct device
    image_features = image_features.to(device)
    
    # Encode text
    text_features = model.from_text_to_embeddings(text_prompts, device)  # [num_prompts, dim]
    
    # Calculate similarity scores following CONCH implementation
    # sim_scores = (image_embedings @ text_embedings.T * model.logit_scale.exp()).softmax(dim=-1)
    logit_scale = model.model.logit_scale.exp()
    
    # Calculate similarity matrix and multiply by logit_scale
    similarity = torch.matmul(image_features, text_features.T) * logit_scale  # [num_images, num_prompts]
    
    # Calculate predictions
    predictions = similarity.argmax(dim=1)
    
    # Calculate softmax probabilities (using model's logit_scale instead of fixed temperature)
    probs = F.softmax(similarity, dim=-1)
    
    results = {
        "predictions": predictions.detach().cpu().numpy(),
        "probabilities": probs.detach().cpu().numpy(),
        "similarities": similarity.detach().cpu().numpy()
    }
    
    return results


def evaluate_zero_shot_dataset(model_name: str,
                              image_features: torch.Tensor,
                              labels: torch.Tensor,
                              text_prompts: List[str],
                              class_names: List[str],
                              device: str,
                              batch_size: int = 32) -> Dict:
    """
    Evaluate zero-shot performance on entire dataset - directly use extracted image features
    
    Args:
        model_name: Model name (for model_factory, should have from_text_to_embeddings method and model.logit_scale attribute)
        image_features: Extracted image features [num_images, dim]
        labels: Image labels [num_images]
        text_prompts: List of complete text prompts, each element corresponds to a class
        class_names: List of class names (for result reporting)
        device: Device
        batch_size: Batch size (for text encoding)
        
    Returns:
        Evaluation result dictionary
    """
    print(f"Starting Zero-shot inference test with model: {model_name}")
    print(f"Number of classes: {len(class_names)}, Number of prompts: {len(text_prompts)}")
    
    # Verify prompt count matches class count
    if len(text_prompts) != len(class_names):
        raise ValueError(f"Number of prompts ({len(text_prompts)}) does not match number of classes ({len(class_names)})")
    
    # Use model_factory to create model (should have from_text_to_embeddings method and model.logit_scale)
    model = encoder_factory(model_name)
    model.to(device)
    model.eval()

    # Directly use extracted image features for zero-shot classification
    print(f"Starting zero-shot evaluation, image features shape: {image_features.shape}")
    
    probabilities = model.run_zero_shot(text_prompts, image_features, device)
    
    results = {
        "labels": labels.cpu().numpy(),
        "probabilities": probabilities.cpu().numpy()}
    
    return results

