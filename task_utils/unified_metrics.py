"""
Unified metrics calculation and saving system
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, 
    f1_score, recall_score, precision_score, cohen_kappa_score,
    confusion_matrix
)
from typing import Dict, Any, Union, List, Optional, Tuple
import json


def cal_scores(probs: Union[torch.Tensor, np.ndarray],
               labels: Union[torch.Tensor, np.ndarray],
               num_classes: int) -> Dict[str, Any]:
    """
    Calculate unified evaluation metrics
    
    Args:
        probs: Model output probabilities [batch_size, num_classes]
        labels: True labels [batch_size]
        num_classes: Number of classes
    
    Returns:
        Dict: Dictionary containing all evaluation metrics
    """
    # Convert to torch tensor
    if isinstance(probs, torch.Tensor):
        probs = probs.clone().detach()
    else:
        probs = torch.tensor(probs, dtype=torch.float32)
    
    if isinstance(labels, torch.Tensor):
        labels = labels.clone().detach()
    else:
        labels = torch.tensor(labels, dtype=torch.long)
    
    # Get predicted classes
    predicted_classes = torch.argmax(probs, dim=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels.numpy(), predicted_classes.numpy())
    
    # Check if already probabilities (row sums ≈ 1), if not perform softmax normalization
    row_sums = probs.sum(dim=1)
    is_probability = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    
    if not is_probability:
        # Input is logits, need softmax to convert to probabilities
        probs = F.softmax(probs, dim=1)
    # Otherwise already probabilities, use directly
    
    # Calculate AUC
    if num_classes > 2:
        macro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), 
                                 average='macro', multi_class='ovr')
        micro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), 
                                 average='micro', multi_class='ovr')
        weighted_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), 
                                    average='weighted', multi_class='ovr')
    else:
        macro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs[:,1].numpy())
        weighted_auc = micro_auc = macro_auc
    
    # Calculate F1 score
    weighted_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
    macro_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='macro')
    micro_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='micro')
    
    # Calculate recall
    weighted_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
    macro_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='macro')
    micro_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='micro')
    
    # Calculate precision
    weighted_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
    macro_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='macro')
    micro_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='micro')
    
    # Calculate balanced accuracy
    baccuracy = balanced_accuracy_score(labels.numpy(), predicted_classes.numpy())
    
    # Calculate Kappa coefficient
    quadratic_kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='quadratic')
    linear_kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='linear')
    
    # Calculate confusion matrix
    confusion_mat = confusion_matrix(labels.numpy(), predicted_classes.numpy())
    
    # Organize metrics dictionary
    metrics = {
        'acc': accuracy,
        'bacc': baccuracy,
        'macro_auc': macro_auc,
        'micro_auc': micro_auc,
        'weighted_auc': weighted_auc,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'macro_recall': macro_recall,
        'micro_recall': micro_recall,
        'weighted_recall': weighted_recall,
        'macro_pre': macro_precision,
        'micro_pre': micro_precision,
        'weighted_pre': weighted_precision,
        'quadratic_kappa': quadratic_kappa,
        'linear_kappa': linear_kappa,
        'confusion_mat': confusion_mat
    }
    
    return metrics


class UnifiedMetricsSaver:
    """
    Unified metrics saver
    """
    
    def __init__(self, save_dir: str, task_name: str):
        """
        Initialize saver
        
        Args:
            save_dir: Save directory
            task_name: Task name
        """
        self.save_dir = save_dir
        self.task_name = task_name
        os.makedirs(save_dir, exist_ok=True)
    
    def save_metrics(self, 
                    probs: Union[torch.Tensor, np.ndarray],
                    labels: Union[torch.Tensor, np.ndarray],
                    num_classes: int,
                    additional_info: Optional[Dict[str, Any]] = None,
                    img_names: Optional[List[str]] = None) -> Tuple[str, str]:
        """
        Save metrics and detailed results
        
        Args:
            probs: Model output probabilities
            labels: True labels
            num_classes: Number of classes
            additional_info: Additional information (e.g., task-specific parameters)
            img_names: List of image names (optional)
        
        Returns:
            Tuple[str, str]: (metrics file path, detailed results file path)
        """
        # Calculate metrics
        metrics = cal_scores(probs, labels, num_classes)
        
        # Convert to numpy array for processing
        probs_np = probs.numpy() if isinstance(probs, torch.Tensor) else probs
        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        
        # Calculate predicted results
        predicted_classes = np.argmax(probs_np, axis=1)
        
        # Check if already probabilities, if not normalize
        row_sums = probs_np.sum(axis=1)
        is_probability = np.allclose(row_sums, np.ones_like(row_sums), atol=1e-3)
        
        if not is_probability:
            # Input is logits, need softmax to convert to probabilities
            probs_normalized = F.softmax(torch.tensor(probs_np), dim=1).numpy()
        else:
            # Already probabilities, use directly
            probs_normalized = probs_np
        
        # Prepare metrics data
        metrics_data = {
            'metric': list(metrics.keys()),
            'value': [metrics[key] if not isinstance(metrics[key], np.ndarray) else str(metrics[key].tolist()) 
                     for key in metrics.keys()]
        }
        
        if additional_info:
            for key, value in additional_info.items():
                metrics_data['metric'].append(key)
                metrics_data['value'].append(str(value))
        
        
        detailed_data = {
            'true_label': labels_np.tolist(),
            'predicted_label': predicted_classes.tolist(),
            'probabilities': [probs_normalized[i].tolist() for i in range(len(probs_normalized))]
        }
        
        if img_names is not None:
            if len(img_names) != len(labels_np):
                print(f"Warning: img_names length ({len(img_names)}) does not match labels length ({len(labels_np)}), will not add img_name column")
            else:
                detailed_data['img_name'] = img_names
        
        if img_names is not None and len(img_names) == len(labels_np):
            detailed_df = pd.DataFrame({
                'img_name': detailed_data['img_name'],
                'true_label': detailed_data['true_label'],
                'predicted_label': detailed_data['predicted_label'],
                'probabilities': detailed_data['probabilities']
            })
        else:
            detailed_df = pd.DataFrame(detailed_data)
        
        detailed_path = os.path.join(self.save_dir, f'{self.task_name}_detailed_results.csv')
        detailed_df.to_csv(detailed_path, index=False, encoding='utf-8')
            
        
        
        json_data = {
            'task_name': self.task_name,
            'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                       for k, v in metrics.items() if k != 'confusion_mat'},
            'confusion_matrix': metrics['confusion_mat'].tolist(),
            'num_samples': len(labels_np),
            'num_classes': num_classes,
            'additional_info': additional_info or {}
        }
        
        json_path = os.path.join(self.save_dir, f'{self.task_name}_complete_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return json_path, detailed_path

    def save_few_shot_metrics(self, probs_all_episodes: List[torch.Tensor], targets_all_episodes: List[torch.Tensor], way: int):
        """
        Save Few-shot metrics and detailed results
        
        Args:
            probs_all_episodes: Prediction probabilities for all episodes List of [test_nums, num_classes]
            targets_all_episodes: True labels for all episodes List of [test_nums]
            way: Number of valid classes in each episode
        """
        all_episode_metrics = []
        all_probs_mapped = []
        all_targets_mapped = []
        
        for episode_idx, (probs_episode, targets_episode) in enumerate(zip(probs_all_episodes, targets_all_episodes)):
            # probs_episode: [test_nums, num_classes]
            # targets_episode: [test_nums]
            
            # Find valid classes in this episode (unique classes appearing in targets)
            unique_classes = torch.unique(targets_episode).cpu().numpy()
            
            # Ensure number of valid classes equals way
            assert len(unique_classes) == way, f"Episode {episode_idx}: found {len(unique_classes)} classes, but expected {way} classes"
            
            # Extract probabilities for valid classes
            probs_valid = probs_episode[:, unique_classes]  # [test_nums, way]
            
            # Create class mapping: map original class labels to 0, 1, ..., way-1
            class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
            
            # Map targets to new label space
            targets_mapped = torch.tensor([class_mapping[int(t)] for t in targets_episode.cpu().numpy()])
            
            # Calculate metrics for this episode
            episode_metrics = cal_scores(probs_valid, targets_mapped, way)
            all_episode_metrics.append(episode_metrics)
            
            # Save mapped data for overall evaluation
            all_probs_mapped.append(probs_valid)
            all_targets_mapped.append(targets_mapped)
        
        # Aggregate metrics for all episodes (only calculate mean and std)
        aggregated_metrics = {}
        metric_keys = ['acc', 'bacc', 'macro_auc', 'micro_auc', 'weighted_auc', 
                      'macro_f1', 'micro_f1', 'weighted_f1',
                      'macro_recall', 'micro_recall', 'weighted_recall',
                      'macro_pre', 'micro_pre', 'weighted_pre',
                      'quadratic_kappa', 'linear_kappa']
        
        for key in metric_keys:
            values = [metrics[key] for metrics in all_episode_metrics]
            aggregated_metrics[f'{key}_mean'] = np.mean(values)
            aggregated_metrics[f'{key}_std'] = np.std(values)
        
        # Concatenate data from all episodes for computing confusion matrix
        all_probs_concat = torch.cat(all_probs_mapped, dim=0)
        all_targets_concat = torch.cat(all_targets_mapped, dim=0)
        
        # Calculate overall confusion matrix
        predicted_classes = torch.argmax(all_probs_concat, dim=1).numpy()
        overall_confusion_mat = confusion_matrix(all_targets_concat.numpy(), predicted_classes)
        
        # 1. Save metrics for each episode (JSON format, including confusion matrix)
        episode_metrics_path = os.path.join(self.save_dir, f'{self.task_name}_per_episode_metrics.json')
        per_episode_data = []
        
        for ep_idx, ep_metrics in enumerate(all_episode_metrics):
            episode_dict = {'episode': ep_idx}
            # Add all metrics
            for key in metric_keys:
                episode_dict[key] = float(ep_metrics[key])
            # Add confusion matrix for this episode
            episode_dict['confusion_matrix'] = ep_metrics['confusion_mat'].tolist()
            per_episode_data.append(episode_dict)
        
        with open(episode_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(per_episode_data, f, indent=2, ensure_ascii=False)
        
        # 2. Save aggregated results (including mean, std, and overall confusion matrix)
        json_data = {
            'task_name': self.task_name,
            'num_episodes': len(probs_all_episodes),
            'way': way,
            'aggregated_metrics': {k: float(v) for k, v in aggregated_metrics.items()},
            'overall_confusion_matrix': overall_confusion_mat.tolist()
        }
        
        json_path = os.path.join(self.save_dir, f'{self.task_name}_few_shot_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        
        return json_path, episode_metrics_path

def create_metrics_saver(save_dir: str, task_name: str) -> UnifiedMetricsSaver:
    """
    Convenience function to create metrics saver
    
    Args:
        save_dir: Save directory
        task_name: Task name
    
    Returns:
        UnifiedMetricsSaver: Metrics saver instance
    """
    return UnifiedMetricsSaver(save_dir, task_name)
