import os
import torch
# Must be set before importing any other libraries!
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1' 
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OPENMP'] = 'FALSE'
os.environ['GOTO_NUM_THREADS'] = '1'

# Limit PyTorch threads
torch.set_num_threads(1)
import argparse
import warnings
from task_utils.eval_linear_probe import eval_linear_probe
from task_utils.fewshot import eval_knn,eval_fewshot
from task_utils.common_utils import save_results_as_txt
from task_utils.zero_shot import evaluate_zero_shot_dataset, load_class_names_from_txt
from task_utils.unified_metrics import create_metrics_saver
warnings.filterwarnings("ignore")
import numpy as np


def main(args):
    # Parse TASK string into a list
    TASK = [task.strip() for task in args.TASK.split(',')]
    # Validate task names
    valid_tasks = ['Linear-Probe','KNN','Proto','Few-shot','Zero-shot']
    for task in TASK:
        if task not in valid_tasks:
            raise ValueError(f"Invalid task name: {task}. Valid tasks: {valid_tasks}")
    
    device = args.device
    os.makedirs(args.log_dir,exist_ok=True)
    save_description_path = os.path.join(args.log_dir,'EXP_NAME.txt')
    if args.log_description is not None:
        save_results_as_txt(args.log_description,save_description_path)
    for task in TASK:
        os.makedirs(os.path.join(args.log_dir,task),exist_ok=True)
    if len(TASK) == 0:
        raise ValueError("No task specified")
    # All tasks are based on pre-extracted feature files, no need to create datasets
    train_dataset = None
    val_dataset = None
    test_dataset = None
    # Load from pre-extracted feature files
    if not os.path.exists(args.train_feature_file) or not os.path.exists(args.test_feature_file):
        raise FileNotFoundError(f"Feature files not found: {args.train_feature_file} or {args.test_feature_file}")
    
    print(f"Loading pre-extracted features: {args.train_feature_file}")
    train_assert = torch.load(args.train_feature_file,weights_only=False)
    test_assert = torch.load(args.test_feature_file,weights_only=False)
    
    # Adapt feature format
    if isinstance(train_assert, dict) and 'embeddings' in train_assert:
        # New format: {'embeddings': ..., 'labels': ...}
        train_feats = torch.Tensor(train_assert['embeddings'])
        train_labels = torch.Tensor(train_assert['labels']).type(torch.long)
        test_feats = torch.Tensor(test_assert['embeddings'])
        test_labels = torch.Tensor(test_assert['labels']).type(torch.long)
        # Extract img_names (if exists)
        test_img_names = test_assert.get('img_names', None)
    else:
        # Old format: directly feature tensor
        train_feats = torch.Tensor(train_assert)
        test_feats = torch.Tensor(test_assert)
        test_img_names = None
        # Get labels from dataset (if dataset exists)
        if train_dataset is not None and test_dataset is not None:
            train_labels = torch.Tensor([train_dataset[i][1] for i in range(len(train_dataset))]).type(torch.long)
            test_labels = torch.Tensor([test_dataset[i][1] for i in range(len(test_dataset))]).type(torch.long)
        else:
            raise ValueError("Feature files do not contain label information and no dataset is available. Please use feature files with labels.")
    
    # Process validation set (if exists)
    if val_dataset is not None and hasattr(args, 'val_feature_file') and args.val_feature_file:
        if os.path.exists(args.val_feature_file):
            val_assert = torch.load(args.val_feature_file)
            if isinstance(val_assert, dict) and 'embeddings' in val_assert:
                val_feats = torch.Tensor(val_assert['embeddings'])
                val_labels = torch.Tensor(val_assert['labels']).type(torch.long)
            else:
                val_feats = torch.Tensor(val_assert)
                val_labels = torch.Tensor([val_dataset[i][1] for i in range(len(val_dataset))]).type(torch.long)
        else:
            print("Validation feature file not found, skipping validation set")
            val_feats, val_labels = None, None
    else:
        val_feats, val_labels = None, None
    if 'Linear-Probe' in TASK:
        linprobe_eval_metrics, linprobe_dump = eval_linear_probe(
            train_feats = train_feats,
            train_labels = train_labels,
            valid_feats = val_feats ,
            valid_labels = val_labels,
            test_feats = test_feats,
            test_labels = test_labels,
            device = device,
            max_iter = args.max_iteration,
            use_sklearn = args.use_sklearn,
            verbose= True)
        
        # Use unified metrics saving system
        linear_probe_save_dir = os.path.join(args.log_dir, 'Linear-Probe')
        Linear_probe_metrics_saver = create_metrics_saver(linear_probe_save_dir, 'Linear-Probe')
        
        Linear_probe_metrics_saver.save_metrics(
            probs=linprobe_dump['probs_all'],
            labels=linprobe_dump['targets_all'],
            num_classes=len(np.unique(linprobe_dump['targets_all'])),
            img_names=test_img_names
        )
        
    if 'KNN' in TASK or 'Proto' in TASK:
        knn_eval_metrics, knn_dump, proto_eval_metrics, proto_dump = eval_knn(
            train_feats = train_feats,
            train_labels = train_labels,
            valid_feats = val_feats ,
            valid_labels = val_labels,
            test_feats = test_feats,
            test_labels = test_labels,
            center_feats = True,
            normalize_feats = True,
            n_neighbors = args.n_neighbors,
            device = device)
    
    if 'KNN' in TASK:
        knn_save_dir = os.path.join(args.log_dir, 'KNN')
        # Use unified metrics saving system
        knn_metrics_saver = create_metrics_saver(knn_save_dir, 'KNN')
        knn_metrics_saver.save_metrics(
            probs=knn_dump['probs_all'],
            labels=knn_dump['targets_all'],
            num_classes=len(np.unique(knn_dump['targets_all'])),
            img_names=test_img_names
        )
        
    if 'Proto' in TASK:
        proto_save_dir = os.path.join(args.log_dir, 'Proto')
        proto_metrics_saver = create_metrics_saver(proto_save_dir, 'Proto')
        proto_metrics_saver.save_metrics(
            probs=proto_dump['probs_all'],
            labels=proto_dump['targets_all'],
            num_classes=len(np.unique(proto_dump['targets_all'])),
            img_names=test_img_names
        )
        
    
    if 'Few-shot' in TASK:
        # Get number of classes and samples per class
        unique_labels = torch.unique(train_labels)
        num_classes = len(unique_labels)
        
        # Calculate sample count for each class
        class_counts = {}
        for label in unique_labels:
            class_counts[label.item()] = (train_labels == label).sum().item()
        
        min_samples_per_class = min(class_counts.values())
        print(f"Dataset info: {num_classes} classes, minimum samples: {min_samples_per_class}")
        
        n_way = args.n_way
        ways = [int(way) for way in n_way.split(',')]
        
        # Filter valid way values
        valid_ways = [way for way in ways if way <= num_classes]
        if len(valid_ways) != len(ways):
            skipped_ways = [way for way in ways if way > num_classes]
            print(f"Skipping invalid way values: {skipped_ways} (exceeds number of classes {num_classes})")

        if args.use_all_way:
            valid_ways = list(range(2, num_classes + 1))
        
        for way in valid_ways:
            save_dir = os.path.join(args.log_dir,'Few-shot',f'way_{way}')
            os.makedirs(save_dir,exist_ok=True)
            
            # Parse n_shot parameter
            n_shot_list = [int(shot.strip()) for shot in args.n_shot.split(',')]
            
            # Filter valid shot values
            valid_shots = [shot for shot in n_shot_list if shot <= min_samples_per_class]
            
            for shot in valid_shots:
                fewshot_metrics_saver = create_metrics_saver(save_dir, f'Fewshot_{way}way_{shot}shot')
                print(f"Few-shot evaluation: {way}-way {shot}-shot")
                probs_all, targets_all = eval_fewshot(
                train_feats = train_feats,
                train_labels = train_labels,
                valid_feats = val_feats ,
                valid_labels = val_labels,
                test_feats = test_feats,
                test_labels = test_labels,
                n_iter = args.n_iter, # draw 500 few-shot episodes
                n_way = way, # use all class examples
                n_shot = shot, # 4 examples per class (as we don't have that many)
                n_query = test_feats.shape[0], # evaluate on all test samples
                center_feats = True,
                normalize_feats = True,
                average_feats = True,)
                fewshot_metrics_saver.save_few_shot_metrics(
                    probs_all_episodes=probs_all,
                    targets_all_episodes=targets_all,
                    way=way
                )
                
                
    if 'Zero-shot' in TASK:
        class_names = load_class_names_from_txt(args.class2id_txt)
        
        # Merge features and labels
        combined_feats = [train_feats, val_feats, test_feats] if val_feats is not None else [train_feats, test_feats]
        combined_labels = [train_labels, val_labels, test_labels] if val_feats is not None else [train_labels, test_labels]
        
        # Merge img_names (if exists)
        # Note: Usually only test set has img_names, train and val may not
        if test_img_names is not None:
            # Create placeholders for train and val (if they don't have img_names)
            train_img_names = [f"train_{i}" for i in range(len(train_feats))]
            val_img_names = [f"val_{i}" for i in range(len(val_feats))] if val_feats is not None else []
            combined_img_names = train_img_names + val_img_names + test_img_names if val_feats is not None else train_img_names + test_img_names
        else:
            combined_img_names = None
                
        # Merge all features
        combined_feats = torch.cat(combined_feats, dim=0)
        combined_labels = torch.cat(combined_labels, dim=0)
                
        # Load complete prompts from file (one complete prompt per line for each class)
        with open(args.zeroshot_prompt_file, 'r', encoding='utf-8') as f:
            prompt_content = f.read().strip()
        
        # Parse prompts: one complete prompt per line, ignore comments and empty lines
        text_prompts = [line.strip() for line in prompt_content.split('\n') 
                       if line.strip() and not line.strip().startswith('#')]        
        
        zeroshot_results = evaluate_zero_shot_dataset(
            model_name=args.zeroshot_model_name,  # Use model_factory to create model
            image_features=combined_feats,  # Directly use extracted image features
            labels=combined_labels,  # Directly use extracted labels
            text_prompts=text_prompts,  # Pass parsed prompt list
            class_names=class_names,  # For result reporting
            device=args.device
        )

               
        # Use unified metrics saving system
        zeroshot_save_dir = os.path.join(args.log_dir, 'Zero-shot')
        zeroshot_metrics_saver = create_metrics_saver(zeroshot_save_dir, 'Zero-shot')
        
        # Prepare prediction results and labels
        zeroshot_labels = np.array(zeroshot_results['labels'])
        zeroshot_probabilities = np.array(zeroshot_results['probabilities']) if 'probabilities' in zeroshot_results else None
        
        # Save metrics and detailed results
        zeroshot_metrics_saver.save_metrics(
            probs=zeroshot_probabilities,
            labels=zeroshot_labels,
            num_classes=len(np.unique(zeroshot_labels)),
            img_names=combined_img_names
        )
        


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    General_args = parser.add_argument_group('General')
    Linear_probe_args = parser.add_argument_group('Linear_probe')
    KNN_and_proto_args = parser.add_argument_group('KNN_and_proto')
    Few_shot_args = parser.add_argument_group('Few-shot')
    Zero_shot_args = parser.add_argument_group('Zero-shot')
    # General
    General_args.add_argument('--TASK', type=str, default='Linear-Probe', help='Linear-Probe,KNN,Proto,Few-shot,Zero-shot')
    General_args.add_argument('--class2id_txt', type=str, default='./datasets/CAMEL.txt')
    General_args.add_argument('--train_feature_file', type=str, default='./ROI_Features/Dataset_[CAMEL]_Model_[conch_v1]_Size_[448]_train.pt', help='Path to training features file')
    General_args.add_argument('--test_feature_file', type=str, default='./ROI_Features/Dataset_[CAMEL]_Model_[conch_v1]_Size_[448]_test.pt', help='Path to test features file')
    General_args.add_argument('--val_feature_file', type=str, default=None, help='Path to validation features file (optional)')
    General_args.add_argument('--log_dir', default='./results', help='path where to save')
    General_args.add_argument('--log_description', type=str, default='ROI Benchmarking', help='Experiment description')
    General_args.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device') 
    # Linear_probe
    Linear_probe_args.add_argument('--max_iteration', type=int, default=1000)
    Linear_probe_args.add_argument('--use_sklearn', default=False, help='use sklearn logistic regression')
    # KNN_and_proto
    KNN_and_proto_args.add_argument('--n_neighbors', type=int, default=20)
    # Few_shot
    Few_shot_args.add_argument('--n_iter', type=int, default=100) # Number of episodes
    Few_shot_args.add_argument('--use_all_way', type=bool, default=True) # Whether to use all classes for few-shot
    Few_shot_args.add_argument('--n_way', type=str, default='2,3,4,5,6,7,8,9,10') # Cannot exceed number of classes
    Few_shot_args.add_argument('--n_shot', type=str, default='1,2,4,8,16,32,64,128,256', 
                                help='Number of samples per class, comma-separated') # Number of samples per class    
    # Zero-shot
    Zero_shot_args.add_argument('--zeroshot_model_name', type=str, default='conch_v1', 
                                help='Model name (created using model_factory, should contain encode_text method)')
    Zero_shot_args.add_argument('--zeroshot_prompt_file', type=str, default='./datasets/CAMEL_prompts.txt', 
                                help='Prompt file path (txt format, one complete prompt per line for each class, order matches class2id_txt)')
    Zero_shot_args.add_argument('--zeroshot_batch_size', type=int, default=32, help='Zero-shot batch size')
    Zero_shot_args.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    opt = parser.parse_args()
    main(opt)


 