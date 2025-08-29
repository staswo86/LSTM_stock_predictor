# Stock Price Prediction with Custom LSTM

This project implements a custom Long Short-Term Memory (LSTM) model in PyTorch to forecast next-day **S&P 500** open prices. The goal is to evaluate how window size influences predictive performance while maintaining full control over the recurrent cell implementation.

## Project Structure
- **data/** – raw and processed datasets (not included in repo; downloaded via [yfinance](https://pypi.org/project/yfinance/)).
- **models/** – custom LSTM implementation in PyTorch.
- **notebooks/** – experimental scripts and visualizations.
- **main.py** – training and evaluation pipeline.

## Methodology
- **Data**: Daily open prices of the S&P 500 (`^GSPC`), 2020-01-01 through 2024-12-31, downloaded with *yfinance*.
- **Preprocessing**: 
  - Sliding windows of size 10, 20, and 50 used as input.
  - Min–Max scaling applied to features (recommend fitting on training set only).
- **Model**: Custom single-layer LSTM with 8 hidden units, implemented from scratch using PyTorch tensor operations.
- **Training**: 
  - Optimizer: Adam (lr=0.001), loss: MSE.  
  - Epochs: 4 (extendable).  
  - Batch size: 16.
- **Evaluation**: Test performance reported with MSE and R². Additional metrics (MAE, RMSE) recommended.

## Reproducibility Notes
- **Avoid data leakage**: Fit scalers on training data only; transform validation/test separately.
- **Splitting**: Prefer chronological train/validation/test splits over random splits.
- **Seeds**: Set random seeds for Python, NumPy, and PyTorch for reproducibility.
- **Device**: Ensure model and tensors are moved consistently to CPU/GPU.

## Requirements
- Python ≥ 3.8  
- [PyTorch](https://pytorch.org/)  
- [NumPy](https://numpy.org/)  
- [Pandas](https://pandas.pydata.org/)  
- [yfinance](https://pypi.org/project/yfinance/)  
- [scikit-learn](https://scikit-learn.org/stable/)  
- [matplotlib](https://matplotlib.org/)

Install all dependencies:
```bash
pip install -r requirements.txt
