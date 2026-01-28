"""
Code based on sampler from @mileyan/simple_shot
Adapted from https://github.com/mbanani/lgssl/blob/df45bae647fc24dce8a6329eb697944053e9a8a0/lgssl/evaluation/fewshot.py.
"""


import logging
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import sklearn.neighbors
import torch
from torch.nn.functional import normalize
from torch.utils.data import Sampler
from tqdm import tqdm
import torch.nn.functional as F
from .metrics import get_eval_metrics


def eval_knn(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    valid_feats: torch.Tensor,
    valid_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    combine_trainval: bool = True,
    center_feats: bool = True,
    normalize_feats: bool = True,
    average_feats: bool = True,
    n_neighbors: int = 20,
    num_workers: int = 8,
    device = None,
):
    """
    Evaluate K-Nearest Neighbors (KNN) algorithm for few-shot learning.
    Adapted from https://github.com/mbanani/lgssl/blob/df45bae647fc24dce8a6329eb697944053e9a8a0/lgssl/evaluation/fewshot.py.

    Args:
        train_feats (torch.Tensor): Training features.
        train_labels (torch.Tensor): Training labels.
        test_feats (torch.Tensor): Test features.
        test_labels (torch.Tensor): Test labels.
        center_feats (bool, optional): Whether to center the features. Defaults to True.
        normalize_feats (bool, optional): Whether to normalize the features. Defaults to True.
        average_feats (bool, optional): Whether to compute prototypes by averaging features. Defaults to True.
        n_neighbors (int, optional): Num neighbors to consider in KNN. Defaults to 20.
        num_workers (int, optional): Num workers for parallel processing. Defaults to 8.

    Returns:
        tuple: A tuple containing the following:
            - proto_metrics (dict): Results prototype-based evaluation.
            - proto_dump (dict): Dumped data for prototype-based evaluation.
            - knn_metrics (dict): Results KNN evaluation.
            - knn_dump (dict): Dumped data for KNN evaluation.
    """

    # Get train and test
    if combine_trainval and (valid_feats is not None):
        train_feats = torch.cat([train_feats, valid_feats], dim=0)
        train_labels = torch.cat([train_labels, valid_labels], dim=0)
    
    # Always set source and query features/labels
    feats_source = train_feats
    labels_source = train_labels
    feats_query = test_feats
    labels_query = test_labels
    
    logging.info(f"KNN Evaluation: Train Shape {feats_source.shape}")
    logging.info(f"KNN Evaluation: Test Shape {feats_query.shape}")

    ### Centering features (for each channel dim across samples)
    feats_mean = None
    if center_feats:
        feats_mean = feats_source.mean(dim=0, keepdims=True)
        feats_query = feats_query - feats_mean
        feats_source = feats_source - feats_mean

    ### Normalizing features across channel dim
    if normalize_feats:
        feats_source = normalize(feats_source, dim=-1, p=2)
        feats_query = normalize(feats_query, dim=-1, p=2)

    # Compute prototypes & assert labels are correct cpu-version
    if average_feats:
        # Convert labels to numpy for indexing if needed
        if isinstance(labels_source, torch.Tensor):
            labels_source_np = labels_source.cpu().numpy()
        else:
            labels_source_np = labels_source
        
        unique_labels = sorted(np.unique(labels_source_np))
        feats_proto = torch.vstack(
            [feats_source[torch.where(labels_source == c)[0]].mean(dim=0) for c in unique_labels]
        )
        labels_proto = torch.Tensor(unique_labels)

    # SimpleShot Eval
    pw_dist = (feats_query[:, None] - feats_proto[None, :]).norm(dim=-1, p=2)
    # Convert distance to probability (smaller distance = higher probability, so use negative distance)
    probs_all_proto = F.softmax(-pw_dist, dim=1)
    labels_pred_proto = labels_proto[pw_dist.min(dim=1).indices]
    proto_metrics = get_eval_metrics(labels_query, labels_pred_proto, prefix="proto_")
    
    proto_dump = {
        "preds_all": labels_pred_proto.cpu().numpy() if isinstance(labels_pred_proto, torch.Tensor) else labels_pred_proto,
        "targets_all": labels_query.cpu().numpy() if isinstance(labels_query, torch.Tensor) else labels_query,
        "probs_all": probs_all_proto.cpu().numpy() if isinstance(probs_all_proto, torch.Tensor) else probs_all_proto,
        "proto_feats": feats_proto.cpu().numpy(),
    }
    if feats_mean is not None:
        proto_dump["proto_mean"] = feats_mean.cpu().numpy()

    # KNN Eval - Use PyTorch for faster computation (especially with GPU)
    # Ensure labels are long type for indexing
    if not isinstance(labels_source, torch.Tensor):
        labels_source = torch.tensor(labels_source, dtype=torch.long)
    else:
        labels_source = labels_source.long()
    
    # Ensure tensors are on the same device
    if device is not None:
        if isinstance(device, str):
            device = torch.device(device)
        if isinstance(device, torch.device):
            feats_source = feats_source.to(device)
            feats_query = feats_query.to(device)
            labels_source = labels_source.to(device)
    
    # Use PyTorch for KNN prediction (much faster, especially with GPU)
    batch_size = 10000  # Process in batches to avoid memory issues
    n_query = feats_query.shape[0]
    labels_pred_knn_list = []
    probs_knn_list = []
    
    # Get unique labels and create mapping
    unique_labels = torch.unique(labels_source).cpu().numpy()
    num_classes = len(unique_labels)
    label_to_idx = {int(label): idx for idx, label in enumerate(unique_labels)}
    
    logging.info(f"KNN prediction: processing {n_query} queries in batches of {batch_size}")
    
    for i in range(0, n_query, batch_size):
        end_idx = min(i + batch_size, n_query)
        feats_query_batch = feats_query[i:end_idx]
        
        # Compute pairwise distances: (batch_size, n_source)
        # Using cosine similarity (since features are normalized) is faster than L2 distance
        # But for consistency with sklearn, we use L2 distance
        distances = torch.cdist(feats_query_batch, feats_source, p=2)  # (batch_size, n_source)
        
        # Get k nearest neighbors
        _, topk_indices = torch.topk(distances, k=n_neighbors, dim=1, largest=False)  # (batch_size, k)
        
        # Get labels of k nearest neighbors
        topk_labels = labels_source[topk_indices]  # (batch_size, k)
        
        # Vote for prediction (most common label) - vectorized version
        # Convert labels to indices for bincount
        max_label = int(labels_source.max().item())
        label_to_idx_tensor = torch.zeros(max_label + 1, dtype=torch.long, device=topk_labels.device)
        for label, idx in label_to_idx.items():
            if label <= max_label:
                label_to_idx_tensor[label] = idx
        
        # Convert topk_labels to indices
        topk_label_indices = label_to_idx_tensor[topk_labels.long()]  # (batch_size, k)
        
        # Use bincount for each sample to count votes (much faster)
        batch_preds = []
        batch_probs = []
        
        for j in range(topk_label_indices.shape[0]):
            # Count votes using bincount
            counts = torch.bincount(topk_label_indices[j], minlength=num_classes).float()
            # Get most common label (convert back to original label)
            pred_idx = counts.argmax().item()
            pred_label = unique_labels[pred_idx]
            batch_preds.append(pred_label)
            
            # Create probability distribution
            prob = (counts / n_neighbors).cpu().numpy()
            batch_probs.append(prob)
        
        labels_pred_knn_list.extend(batch_preds)
        probs_knn_list.extend(batch_probs)
        
        if (i // batch_size + 1) % 10 == 0:
            logging.info(f"Processed {end_idx}/{n_query} queries")
    
    # Convert to numpy arrays
    labels_pred_knn = np.array(labels_pred_knn_list)
    probs_knn = np.array(probs_knn_list)
    
    logging.info(f"KNN prediction completed")
    knn_metrics = get_eval_metrics(labels_query, labels_pred_knn, prefix=f"knn{n_neighbors}_")
    knn_dump = {
        "preds_all": labels_pred_knn,
        "targets_all": labels_query.cpu().numpy() if isinstance(labels_query, torch.Tensor) else labels_query,
        "probs_all": probs_knn,  # Save probability predictions
    }

    return knn_metrics, knn_dump, proto_metrics, proto_dump 


def eval_fewshot(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    valid_feats: torch.Tensor,
    valid_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    combine_trainval: bool = True,
    n_iter: int = 1000,
    n_way: int = -1,
    n_shot: int = 256,
    n_query: int = -1,
    center_feats: bool = True,
    normalize_feats: bool = True,
    average_feats: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Evaluate few-shot learning performance.

    Args:
        train_feats (torch.Tensor): Training features.
        train_labels (torch.Tensor): Training labels.
        test_feats (torch.Tensor): Test features.
        test_labels (torch.Tensor): Test labels.
        n_iter (int, optional): Num iterations. Defaults to 1000.
        n_way (int, optional): Num classes per few-shot task. Defaults to -1 (use all classes in test set).
        n_shot (int, optional): Num support examples per class. Defaults to 256 examples per class in train set.
        n_query (int, optional): Num query examples per class. Defaults to -1 (use all examples in test set).
        center_feats (bool, optional): Whether to center the features. Defaults to True.
        normalize_feats (bool, optional): Whether to normalize the features. Defaults to True.
        average_feats (bool, optional): Whether to average the features. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, dict]: A tuple containing the results from every few-shot episode and its mean/std.
    """
    logging.info(
        f"FS Evaluation: n_iter: {n_iter}, n_way: {n_way}, n_shot: {n_shot}, n_query: {n_query}, center_feats: {center_feats}, normalize_feats: {normalize_feats}, average_feats: {average_feats}"
    )
    logging.info(f"FS Evaluation: Train Shape {train_feats.shape}")
    logging.info(f"FS Evaluation: Test Shape {test_feats.shape}")
    if combine_trainval and (valid_feats is not None):
        train_feats = torch.cat([train_feats, valid_feats], dim=0)
        train_labels = torch.cat([train_labels, valid_labels], dim=0)
    if n_way == -1:
        n_way = len(np.unique(train_labels))
        assert n_way == len(np.unique(test_labels))

    if n_query == -1:
        logging.info("Using all test samples for query")

    # Set up sampler
    fewshot_sampler = FewShotEpisodeSampler(
        train_labels,
        test_labels,
        n_iter,
        n_way,
        n_shot,
        n_query,
    )

    # test model on dataset -- really more tasks than batches
    results_all = []
    probs_all = []
    targets_all = []
    n_way = n_way
    n_shot = n_shot
    
    # Determine total number of classes (using union of train and test sets)
    all_labels = torch.cat([train_labels, test_labels])
    num_classes = len(torch.unique(all_labels))

    for task in tqdm(fewshot_sampler):
        source, query = task

        # get train and test
        feats_source = train_feats[source]
        labels_source = train_labels[source]
        if n_query == -1:
            feats_query = test_feats.detach().clone()
            labels_query = test_labels.detach().clone()
        else:
            feats_query = test_feats[query]
            labels_query = test_labels[query]

        # center
        if center_feats:
            feats_mean = feats_source.mean(dim=0, keepdims=True)
            feats_query = feats_query - feats_mean
            feats_source = feats_source - feats_mean

        # normalize
        if normalize_feats:
            feats_source = normalize(feats_source, dim=-1, p=2)
            feats_query = normalize(feats_query, dim=-1, p=2)

        # compute prototypes & assert labels are correct
        if average_feats:
            feats_proto = feats_source.view(n_way, n_shot, -1).mean(dim=1)
            labels_proto = labels_source.view(n_way, n_shot)
            try:
                assert (labels_proto.min(dim=1).values == labels_proto.max(dim=1).values).all()
            except:
                breakpoint()
            labels_proto = labels_proto[:, 0]
        else:
            feats_proto = feats_source
            labels_proto = labels_source

        # classify to prototypes
        pw_dist = (feats_query[:, None] - feats_proto[None, :]).norm(dim=-1, p=2)
        
        # Convert distance to probability (smaller distance = higher probability)
        logits = -pw_dist  # Negative distance as logits
        probs = F.softmax(logits, dim=1)
        
        # Create full probability matrix (corresponding to all classes)
        probs_full = torch.zeros(len(labels_query), num_classes)
        for i, label in enumerate(labels_proto):
            probs_full[:, int(label)] = probs[:, i]
        
        # Save probs and targets for current episode
        probs_all.append(probs_full)
        targets_all.append(labels_query)

    # Return probs and targets for all episodes
    return probs_all, targets_all


class FewShotEpisodeSampler(Sampler):
    """
    Sampler for generating few-shot episodes for training or evaluation.

    Adapted from https://github.com/mbanani/lgssl/blob/df45bae647fc24dce8a6329eb697944053e9a8a0/lgssl/evaluation/fewshot.py.
    """

    def __init__(
        self,
        train_labels: List[int],
        test_labels: List[int],
        n_iter: int,
        n_way: int,
        n_shot: int,
        n_query: int,
    ) -> None:
        """
        Args:
            train_labels (list): List of training labels.
            test_labels (list): List of test labels.
            n_iter (int): Number of iterations (episodes) to generate.
            n_way (int): Number of classes per episode.
            n_shot (int): Number of samples per class in the support set.
            n_query (int): Number of samples per class in the query set.
        """
        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query

        train_labels = np.array(train_labels)
        self.train_ind = []
        self.test_ind = []
        unique = np.unique(train_labels)
        unique = np.sort(unique)
        for i in unique:
            train_ind = np.argwhere(train_labels == i).reshape(-1)
            self.train_ind.append(train_ind)

            test_ind = np.argwhere(test_labels == i).reshape(-1)
            self.test_ind.append(test_ind)

    def __len__(self) -> int:
        return self.n_iter

    def __iter__(self) -> Tuple[Any, Any]:
        for _ in range(self.n_iter):
            batch_gallery = []
            batch_query = []
            classes = torch.randperm(len(self.train_ind))[: self.n_way]
            for c in classes:
                train_c = self.train_ind[c.item()]
                assert len(train_c) >= (self.n_shot), f"{len(train_c)} < {self.n_shot}"
                train_pos = torch.multinomial(torch.ones(len(train_c)), self.n_shot)
                batch_gallery.append(train_c[train_pos])

                test_c = self.test_ind[c.item()]
                if len(test_c) < (self.n_query):
                    logging.info(f"test class has {len(test_c)} ins. (< {self.n_query})")
                    batch_query.append(test_c)
                else:
                    test_pos = torch.multinomial(torch.ones(len(test_c)), self.n_query)
                    batch_query.append(test_c[test_pos])

            if self.n_shot == 1:
                batch_gallery = np.array(batch_gallery)
                batch_query = np.concatenate(batch_query)
            else:
                batch_gallery = np.concatenate(batch_gallery)
                batch_query = np.concatenate(batch_query)

            yield (batch_gallery, batch_query)
