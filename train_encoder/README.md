# Mini GPT Language Model

A lightweight implementation of a GPT-like language model using PyTorch, designed for educational purposes and small-scale training.

## 📊 Dataset Setup

### Download Sample Datasets

Run the dataset downloader to get sample datasets:

```bash
python download_datasets.py
```

This will download:
- **SQuAD sample** (200 Q&A pairs) - Question answering dataset
- **Daily Dialog sample** (150 Q&A pairs) - Conversational dialogue
- **WikiText sample** (100 Q&A pairs) - Wikipedia text
- **Simple QA** (8 Q&A pairs) - Basic test dataset

### Available Datasets

After downloading, you'll have these datasets in the `datasets/` directory:
- `simple_qa.json` - Basic test dataset
- `squad_sample.json` - SQuAD question answering
- `daily_dialog_sample.json` - Daily conversations
- `wikitext_sample.json` - Wikipedia text

## 🐍 Running with Python

### Prerequisites
- Python 3.7+
- PyTorch 2.0.0+
- NumPy 1.21.0+
- Matplotlib 3.5.0+

### Installation

**Install required packages:**
```bash
pip install -r requirements.txt
```

### Usage

**List available datasets:**
```python
import sys
sys.path.append('mini_gpt')
from mini_gpt.main import list_datasets
list_datasets()
```

**Train with a specific dataset:**
```python
import sys
sys.path.append('mini_gpt')
from mini_gpt.main import train_mini_gpt

# Train with simple QA dataset (5 epochs)
train_mini_gpt("simple_qa.json", epochs=5)

# Train with SQuAD dataset (3 epochs, custom learning rate)
train_mini_gpt("squad_sample.json", epochs=3, learning_rate=0.005)

# Train with Daily Dialog dataset (3 epochs)
train_mini_gpt("daily_dialog_sample.json", epochs=3, learning_rate=0.005)

# Train with custom parameters
train_mini_gpt("simple_qa.json", epochs=10, learning_rate=0.01)
```

OR
# directly update the file name in main.py about the dataset name
python3 main.py

**Make predictions:**
```bash
python predict.py
```

Or in Python:
```python
from predict import predict_text
result = predict_text("how are you")
print(result)
```

### Quick Python Examples

**1. Download datasets and install dependencies:**
```bash
python download_datasets.py
pip install -r requirements.txt
```

**2. List available datasets:**
```python
import sys
sys.path.append('mini_gpt')
from mini_gpt.main import list_datasets
list_datasets()
```

**3. Train with simple QA:**
```python
import sys
sys.path.append('mini_gpt')
from mini_gpt.main import train_mini_gpt
train_mini_gpt("simple_qa.json", epochs=5)
```

**4. Train with SQuAD:**
```python
import sys
sys.path.append('mini_gpt')
from mini_gpt.main import train_mini_gpt
train_mini_gpt("squad_sample.json", epochs=3, learning_rate=0.005)
```

**5. Make predictions:**
```bash
python predict.py
```

## 🏗️ Project Structure

```
train_encoder/
├── mini_gpt/                 # Main model code
│   ├── config.py            # Configuration and hyperparameters
│   ├── data_utils.py        # Data preprocessing and vocabulary
│   ├── models.py            # Transformer model implementation
│   ├── training.py          # Training and inference functions
│   ├── plotting.py          # Training visualization
│   └── main.py              # Main training script
├── datasets/                # Dataset files
├── saved_model/             # Trained models (auto-created)
├── training_plots/          # Training plots (auto-created)
├── download_datasets.py     # Dataset downloader
├── predict.py               # Model prediction script
└── requirements.txt         # Python dependencies
```

## 🎯 Model Architecture

- **Embedding Size**: 128 dimensions
- **Layers**: 2 transformer layers
- **Attention Heads**: 4 multi-head attention
- **Dropout**: 0.1
- **Learning Rate**: 0.01 (configurable)
- **Epochs**: 20 (configurable)

## 📈 Training Features

- **Autoregressive Training**: Generates text token by token
- **Gradient Tracking**: Monitors gradient norms during training
- **Loss Visualization**: Automatic plot generation
- **Model Saving**: Saves trained models and vocabulary
- **Dataset Flexibility**: Easy switching between different datasets

## 📊 Training Plots

The training automatically generates these plots in `training_plots/`:
- `training_loss.png` - Epoch loss progression
- `loss_comparison.png` - Step vs epoch loss comparison
- `gradient_norms.png` - Gradient norm tracking
- `training_summary.png` - Training summary dashboard

## 🎉 Complete Workflow Example

```bash
# 1. Download datasets
python download_datasets.py

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train with simple QA
python -c "import sys; sys.path.append('mini_gpt'); from mini_gpt.main import train_mini_gpt; train_mini_gpt('simple_qa.json', epochs=5)"

# 4. Make predictions
python predict.py
```

## 🔍 Troubleshooting

### Python Issues
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.7+ required)
- Verify dataset files exist in `datasets/` directory

### Training Issues
- Reduce learning rate if loss doesn't converge
- Increase epochs for better results
- Check dataset format (should be JSON with Q&A pairs)

### Memory Issues
- Reduce embedding size in `config.py`
- Use smaller datasets
- Reduce batch size if needed

## 📝 License

This project is for educational purposes. Feel free to modify and experiment! 