# 🔬HistoROIBench

A comprehensive benchmarking framework for evaluating foundation models on histopathology ROI datasets.

## 📋 Overview

HistoROIBench provides a standardized evaluation pipeline for testing various pretrained models on pathology image ROI classification tasks. The framework supports multiple state-of-the-art pathology image encoder models and offers a complete workflow from feature extraction to multi-task evaluation.

### Key Features

- 🎯 **Multi-Model Support**: Integrates 20+ state-of-the-art pathology image encoders
- 🔬 **Multi-Task Evaluation**: Supports 5 different evaluation paradigms (Linear Probe, KNN, Proto, Few-shot, Zero-shot)
- 🚀 **Efficient Pipeline**: Pre-extract features to avoid redundant computation
- 📊 **Unified Metrics**: Standardized evaluation metrics output for easy model comparison
- 🛠️ **Easy Extension**: Modular design for easy addition of new models

## 🤖 Supported Models

The framework supports 20+ pretrained models:

### Mainstream Pathology Models
- **CONCH** (v1, v1.5) - Contrastive learning pathology model
- **UNI** (v1, v2) - Universal pathology foundation model
- **CTransPath** - Transformer-based pathology model
- **Phikon** (v1, v2) - Large-scale pathology pretrained model
- **Virchow** (v1, v2) - Pathology Vision Transformer
- **GigaPath** - Ultra-large-scale pathology model
- **H-optimus** (0, 1) - Optimized pathology encoders
- **Hibou-L** - Large pathology model
- **MUSK** - Multi-scale pathology feature extractor

### Kaiko Series
- kaiko-vitb8, kaiko-vitb16
- kaiko-vits8, kaiko-vits16
- kaiko-vitl14

### Lunit Series
- lunit-vits8

### General Vision Models
- **ResNet50** - Classic convolutional neural network

> **Note**: Configure the corresponding model weight paths in `model_utils/model_weights.json` before use.

## 📝 Supported Tasks

The framework supports the following 5 evaluation tasks:

### 1. Linear Probe
Train a linear classifier on frozen feature extractor to evaluate feature quality.

**Use Cases**:
- Evaluate discriminative power of pretrained features
- Quick model selection

### 2. KNN (K-Nearest Neighbors)
Classification using K-nearest neighbors algorithm without training.

**Use Cases**:
- Evaluate feature clustering performance
- Non-parametric evaluation

### 3. Proto (Prototypical Networks)
Classification based on class prototypes (centroids).

**Use Cases**:
- Few-shot learning scenarios
- Class-balanced evaluation

### 4. Few-shot Learning
Simulate few-shot learning scenarios to test model generalization capability.

**Use Cases**:
- Data-scarce scenarios
- N-way K-shot evaluation

### 5. Zero-shot Learning
Zero-shot classification using text-image alignment capabilities (requires multimodal support).

**Use Cases**:
- Open-vocabulary classification
- Cross-domain generalization evaluation

## 🚀 Usage

### Workflow

```
Dataset Preparation → Feature Extraction → Multi-Task Evaluation → Result Analysis
```

### Step 1: Feature Extraction

Use `00-ROI_Feature_Extract.py` to extract image features from datasets.

#### Parameter Description

**Dataset Parameters:**
```bash
--dataset_split_csv    # Path to dataset split CSV file (required)
                       # CSV format should include: image path, label, split(train/test)
--class2id_txt        # Path to class-to-ID mapping file (required)
                       # Format: one class name per line or "id:class_name"
--dataset_name        # Dataset name for saving feature files
```

**Model Parameters:**
```bash
--model_name          # Model name (see supported models list)
--resize_size         # Image resize size, default: 448
```

**Inference Parameters:**
```bash
--batch_size          # Batch size, default: 256
--num_workers         # Number of data loading workers, default: 8
--device              # Device ID, e.g., 'cuda:0' or 'cpu'
```

**Save Path:**
```bash
--save_dir            # Directory path to save features
```

#### Usage Examples

```bash
python 00-ROI_Feature_Extract.py \
    --dataset_split_csv /path/to/dataset.csv \
    --class2id_txt /path/to/classes.txt \
    --dataset_name CAMEL \
    --model_name conch_v1 \
    --resize_size 448 \
    --batch_size 256 \
    --num_workers 8 \
    --device cuda:0 \
    --save_dir ./ROI_Features
```

**Output Files:**
- `Dataset_[dataset_name]_Model_[model_name]_Size_[size]_train.pt`
- `Dataset_[dataset_name]_Model_[model_name]_Size_[size]_test.pt`

---

### Step 2: Benchmarking

Use `01-ROI_BenchMark_Main.py` to run various evaluation tasks.

#### Parameter Description

**General Parameters:**
```bash
--TASK                # Task list, comma-separated (required)
                      # Options: Linear-Probe,KNN,Proto,Few-shot,Zero-shot
--class2id_txt        # Path to class mapping file
--train_feature_file  # Path to training feature file (required)
--test_feature_file   # Path to test feature file (required)
--val_feature_file    # Path to validation feature file (optional)
--log_dir             # Directory to save results
--log_description     # Experiment description
--device              # Computing device, default: cuda (if available)
```

**Linear Probe Parameters:**
```bash
--max_iteration       # Maximum iterations, default: 1000
--use_sklearn         # Use sklearn's logistic regression, default: False
```

**KNN & Proto Parameters:**
```bash
--n_neighbors         # Number of neighbors for KNN, default: 20
```

**Few-shot Parameters:**
```bash
--n_iter              # Number of few-shot episodes, default: 100
--use_all_way         # Use all classes, default: True
--n_way               # N-way settings, comma-separated, default: '2,3,4,5,6,7,8,9,10'
--n_shot              # K-shot settings, comma-separated
                      # Default: '1,2,4,8,16,32,64,128,256'
```

**Zero-shot Parameters:**
```bash
--zeroshot_model_name    # Model name for zero-shot (must support text encoding)
--zeroshot_prompt_file   # Path to prompt file
                         # Format: one complete prompt per line for each class
--zeroshot_batch_size    # Batch size, default: 32
--num_workers            # Number of data loader workers, default: 4
```

#### Usage Examples

**1. Run Single Task:**
```bash
python 01-ROI_BenchMark_Main.py \
    --TASK Linear-Probe \
    --train_feature_file ./ROI_Features/Dataset_[CAMEL]_Model_[conch_v1]_Size_[448]_train.pt \
    --test_feature_file ./ROI_Features/Dataset_[CAMEL]_Model_[conch_v1]_Size_[448]_test.pt \
    --class2id_txt /path/to/classes.txt \
    --log_dir ./results \
    --device cuda:0
```

**2. Run Multiple Tasks:**
```bash
python 01-ROI_BenchMark_Main.py \
    --TASK Linear-Probe,KNN,Proto,Few-shot \
    --train_feature_file ./ROI_Features/train.pt \
    --test_feature_file ./ROI_Features/test.pt \
    --class2id_txt /path/to/classes.txt \
    --log_dir ./results \
    --n_neighbors 20 \
    --n_way 2,3,4,5 \
    --n_shot 1,2,4,8,16 \
    --device cuda:0
```

**3. Run Zero-shot Task:**
```bash
python 01-ROI_BenchMark_Main.py \
    --TASK Zero-shot \
    --train_feature_file ./ROI_Features/train.pt \
    --test_feature_file ./ROI_Features/test.pt \
    --class2id_txt /path/to/classes.txt \
    --zeroshot_model_name conch_v1 \
    --zeroshot_prompt_file /path/to/prompts.txt \
    --log_dir ./results \
    --device cuda:0
```

**4. Run All Tasks:**
```bash
python 01-ROI_BenchMark_Main.py \
    --TASK Linear-Probe,KNN,Proto,Few-shot,Zero-shot \
    --train_feature_file ./ROI_Features/train.pt \
    --test_feature_file ./ROI_Features/test.pt \
    --class2id_txt /path/to/classes.txt \
    --zeroshot_model_name conch_v1 \
    --zeroshot_prompt_file /path/to/prompts.txt \
    --log_dir ./results \
    --max_iteration 1000 \
    --n_neighbors 20 \
    --n_way 2,3,4,5 \
    --n_shot 1,2,4,8,16 \
    --device cuda:0
```

#### Output Results

Evaluation results will be saved in the directory specified by `--log_dir`, organized by task type:

```
log_dir/
├── EXP_NAME.txt                    # Experiment description
├── Linear-Probe/
│   ├── Linear-Probe_metrics.txt    # Evaluation metrics
│   ├── Linear-Probe_predictions.csv # Prediction results
│   └── ...
├── KNN/
│   ├── KNN_metrics.txt
│   └── ...
├── Proto/
│   ├── Proto_metrics.txt
│   └── ...
├── Few-shot/
│   ├── way_2/
│   │   ├── Fewshot_2way_1shot_metrics.txt
│   │   ├── Fewshot_2way_2shot_metrics.txt
│   │   └── ...
│   └── ...
└── Zero-shot/
    ├── Zero-shot_metrics.txt
    └── ...
```

Each task's metrics file contains:
- Accuracy
- Balanced Accuracy
- Precision, Recall, F1 Score
- Confusion Matrix
- ROC-AUC (if applicable)
- Detailed per-class performance metrics

## 📁 Data Format

This framework provides complete example datasets in the `example_dataset/` directory, demonstrating the required file formats and structure.

### Example Dataset

The example dataset is based on CRC-100K dataset and includes the following files:

```
example_dataset/
├── CRC-100K.csv                      # Dataset split file
├── CRC-100K.txt                      # Class mapping file
└── CRC-100K-Zero_Shot_Prompts.txt   # Zero-shot prompt file
```

### Dataset CSV Format

The dataset split CSV file should contain the following columns:

**Format 1: Standard Format (Recommended)**
```csv
image_path,label,split
/path/to/image1.png,0,train
/path/to/image2.png,1,train
/path/to/image3.png,0,test
...
```

**Format 2: Train/Validation/Test Separated Format**
```csv
train_path,train_label,val_path,val_label,test_path,test_label
/path/to/train1.tif,8,,,/path/to/test1.tif,8.0
/path/to/train2.tif,8,,,/path/to/test2.tif,8.0
...
```

> **Example**: See `example_dataset/CRC-100K.csv`

**Description**:
- `image_path` / `train_path` / `test_path`: Absolute or relative path to image files
- `label` / `train_label` / `test_label`: Class label (integer, starting from 0)
- `split`: Dataset split identifier (`train` / `val` / `test`)
- Empty columns are used as placeholders (e.g., when validation set is empty)

### Class Mapping File Format

`class2id_txt` file format (both formats supported):

**Format 1: ID:ClassName**
```
0:class_name_1
1:class_name_2
2:class_name_3
```

**Format 2: ClassName,ID (CSV format, Recommended)**
```
class_name_1,0
class_name_2,1
class_name_3,2
```

> **Example**: See `example_dataset/CRC-100K.txt`
> ```
> NORM,0
> STR,1
> TUM,2
> MUS,3
> MUC,4
> LYM,5
> DEB,6
> BACK,7
> ADI,8
> ```

**Description**:
- Class names can be abbreviations (e.g., `TUM`) or full names (e.g., `Tumor`)
- IDs must be consecutive integers starting from 0
- Order must correspond to the labels in the dataset CSV

### Zero-shot Prompt File Format

One complete prompt per line corresponding to each class, in the same order as `class2id_txt`:

```
A histopathology image showing class_name_1
A histopathology image showing class_name_2
A histopathology image showing class_name_3
```

> **Example**: See `example_dataset/CRC-100K-Zero_Shot_Prompts.txt`
> ```
> This is a pathology image showing normal tissue characteristics
> This is a pathology image showing stroma tissue characteristics
> This is a pathology image showing tumor tissue characteristics
> This is a pathology image showing muscle tissue characteristics
> This is a pathology image showing mucosa tissue characteristics
> This is a pathology image showing lymphocytes tissue characteristics
> This is a pathology image showing debris tissue characteristics
> This is a pathology image showing background tissue characteristics
> This is a pathology image showing adipose tissue characteristics
> ```

**Description**:
- Each line corresponds to one class, order must exactly match the class mapping file
- Recommend using descriptive prompts that include tissue type characteristics
- Empty lines and comment lines starting with `#` will be ignored
- Prompt quality directly affects zero-shot performance; domain-specific terminology is recommended

## ⚙️ Configure Model Weights

Before use, configure model weight paths in `model_utils/model_weights.json`:

```json
{
    "conch_v1": "/path/to/conch_v1/pytorch_model.bin",
    "uni_v1": "/path/to/uni_v1/weights.pth",
    "phikon": "/path/to/phikon/checkpoint.pth",
    ...
}
```

If a model's weight path is an empty string `""`, the framework will attempt to automatically download from Hugging Face Hub (requires internet connection).

## 📦 Dependencies

```bash
torch
torchvision
numpy
scikit-learn
pandas
tqdm
```

Recommended: Python 3.8+ and PyTorch 1.10+

## 🔧 Project Structure

```
HistoROIBench/
├── 00-ROI_Feature_Extract.py        # Feature extraction script
├── 01-ROI_BenchMark_Main.py         # Main benchmarking script
├── model_utils/                      # Model utilities
│   ├── model_factory.py             # Model factory
│   ├── model_weights.json           # Model weights configuration
│   └── model_zoo/                   # Model implementations
├── dataset_utils/                    # Dataset utilities
│   └── roi_dataset.py               # ROI dataset class
├── task_utils/                       # Task evaluation utilities
│   ├── eval_linear_probe.py         # Linear Probe evaluation
│   ├── fewshot.py                   # Few-shot evaluation
│   ├── zero_shot.py                 # Zero-shot evaluation
│   ├── unified_metrics.py           # Unified metrics saver
│   └── common_utils.py              # Common utility functions
├── README.md                         # Chinese README
└── README_EN.md                      # This file (English README)
```

## 💡 Best Practices

1. **Choose Appropriate Image Size**: Different models have different recommended input sizes (224, 448, etc.), refer to the original model papers
2. **Adjust Batch Size**: Adjust `--batch_size` according to GPU memory; feature extraction can use larger batches
3. **Few-shot Parameter Settings**: Ensure `--n_shot` does not exceed the minimum class sample count
4. **Zero-shot Model Selection**: Only multimodal models (e.g., CONCH) can perform zero-shot evaluation
5. **Multi-task Evaluation**: Recommend extracting features first, then running multiple tasks on the same feature files to save time

## 🔬 Example Workflow

Here's a complete example workflow:

**Step 1: Extract Features**
```bash
# Extract features using CONCH v1
python 00-ROI_Feature_Extract.py \
    --dataset_split_csv ./datasets/CAMEL.csv \
    --class2id_txt ./datasets/CAMEL.txt \
    --dataset_name CAMEL \
    --model_name conch_v1 \
    --resize_size 448 \
    --batch_size 128 \
    --device cuda:0 \
    --save_dir ./features
```

**Step 2: Run Comprehensive Evaluation**
```bash
# Run all evaluation tasks
python 01-ROI_BenchMark_Main.py \
    --TASK Linear-Probe,KNN,Proto,Few-shot,Zero-shot \
    --train_feature_file ./features/Dataset_[CAMEL]_Model_[conch_v1]_Size_[448]_train.pt \
    --test_feature_file ./features/Dataset_[CAMEL]_Model_[conch_v1]_Size_[448]_test.pt \
    --class2id_txt ./datasets/CAMEL.txt \
    --zeroshot_model_name conch_v1 \
    --zeroshot_prompt_file ./datasets/CAMEL_prompts.txt \
    --log_dir ./results/CAMEL_conch_v1 \
    --log_description "Comprehensive evaluation of CONCH v1 on CAMEL dataset" \
    --device cuda:0
```

**Step 3: Analyze Results**
```bash
# Results will be saved in ./results/CAMEL_conch_v1/
# Each task folder contains detailed metrics and predictions
```

## 🎯 Evaluation Tasks Explained

### Linear Probe
Freezes the feature extractor and trains only a linear classifier. This task evaluates how linearly separable the learned features are.

**Advantages**: Fast, simple, good indicator of feature quality  
**Metrics**: Accuracy, Balanced Accuracy, F1 Score, Confusion Matrix

### KNN
Non-parametric classification using K-nearest neighbors in feature space.

**Advantages**: No training required, intuitive  
**Metrics**: Accuracy, Balanced Accuracy, per-class performance

### Prototypical Networks
Classifies samples based on distance to class centroids (prototypes).

**Advantages**: Works well with imbalanced datasets, interpretable  
**Metrics**: Accuracy, Balanced Accuracy, prototype distances

### Few-shot Learning
Evaluates generalization with limited labeled samples (N-way K-shot).

**Advantages**: Tests model robustness in low-data regimes  
**Metrics**: Average accuracy across episodes, standard deviation

### Zero-shot Learning
Classifies without any training examples using text descriptions.

**Advantages**: No labeled data needed, open-vocabulary capability  
**Requirements**: Model must support text encoding (multimodal)  
**Metrics**: Accuracy, per-class performance

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 🙏 Acknowledgements

This project benefits from the following excellent open-source projects and tools:

- **[TRIDENT](https://github.com/mahmoodlab/TRIDENT)** - A toolkit for large-scale whole-slide image processing developed by Mahmood Lab, which provided inspiration and reference for our work
- **[Timm](https://github.com/huggingface/pytorch-image-models)** - PyTorch Image Models library
- **[HuggingFace](https://huggingface.co/)** - Platform for hosting and distributing pretrained models
- All authors and contributors of the open-source models

We thank the community for their continuous contributions and support!

## 📄 License

This project follows the licenses of the respective models. Please ensure compliance with each model's usage terms before use.

## 📚 Citation

If you use this framework in your research, please cite the relevant model papers and this repository.

## 📧 Contact

For questions or suggestions, please submit an Issue or contact the project maintainers.

---

**Happy Benchmarking! 🎉**

