# 🧠 Deep Learning Projects

Collection of deep learning projects covering computer vision, NLP, and PyTorch implementations.

---

## 📁 Projects

### 🖼️ Computer Vision

**🐱🐶 Cat vs Dog Classification** - `cat-vs-dog classification.ipynb`
- Transfer learning with VGG16 (ImageNet weights)
- Binary image classification
- Kaggle dataset integration
- TensorFlow/Keras implementation

**👤 Age & Gender Detection** - `age-gender-revised.ipynb`
- Multi-task learning (predicts age + gender)
- CNN architecture
- TensorFlow/Keras

**👗 Fashion MNIST Classification** - `ann-fashion-mnist-pytorch-gpu.ipynb`
- Multi-class classification (10 fashion categories)
- PyTorch with GPU support
- Artificial Neural Networks

**🏥 Breast Cancer Classification** - `DL Project 1. Breast Cancer Classification with NN and { Using With TensorFlow and also Pytorch }.ipynb`
- Implemented in both TensorFlow and PyTorch
- Binary classification for cancer detection

---

### 🗣️ Natural Language Processing

**📝 Next Word Predictor** - `Next word predictor using LSTM.ipynb`
- LSTM-based sequence prediction
- Text preprocessing and tokenization
- Vocabulary building

**🤔 RNN-based Q&A System** - `pytorch-rnn-based-qa-system.ipynb`
- Question-answering with RNNs
- Custom tokenization
- PyTorch implementation

**🔤 Multi-Class Text Classification** - `git commit multi class classification.ipynb`
- Text classification
- Neural network architecture

---

### 🔥 PyTorch Fundamentals

**🏗️ PyTorch NN Module** - `pytorch-nn-module.ipynb`
- Building neural networks with nn.Module
- Custom architectures
- Forward and backward propagation

**📊 Dataset & DataLoader Demo** - `dataset-and-dataloader-demo.ipynb`
- Custom Dataset classes
- DataLoader for batch processing
- Data pipelines

**🚀 PyTorch Training Pipelines**
- `pytorch-training-pipeline.ipynb`
- `pytorch-training-pipeline-using-dataset-and-dataloader.ipynb`
- Complete training loop implementations
- Model evaluation

---

### ⚙️ Advanced Techniques

**🎯 Keras Tuner Hyperparameter Tuning** - `Keras Tuner Hyperparameter tuning.ipynb`
- Automated hyperparameter search
- Keras Tuner implementation

---

## 🛠️ Requirements

```bash
pip install tensorflow torch torchvision keras keras-tuner pandas numpy matplotlib scikit-learn opencv-python kaggle
```

**Optional:** CUDA for GPU acceleration

---

## 🚀 Getting Started

```bash
# Clone repository
git clone <repository-url>
cd "DL Projects"

# Install dependencies
pip install tensorflow torch torchvision keras pandas numpy matplotlib

# For Kaggle datasets
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Open notebooks in Jupyter or VS Code
```

---

**Total Projects:** 12 notebooks  
**Frameworks:** TensorFlow, Keras, PyTorch  
**Topics:** CNN, RNN, LSTM, Transfer Learning, Hyperparameter Tuning
