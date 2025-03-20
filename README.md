# Data Science Pipeline

A comprehensive data science pipeline that includes data preprocessing, feature engineering, model training, and visualization capabilities.

## Features

- **Data Preprocessing**
  - Missing value handling
  - Outlier detection and handling
  - Feature scaling and normalization
  - Dimensionality reduction

- **Feature Engineering**
  - Text embeddings generation
  - Polynomial feature generation
  - Clustering-based feature generation
  - Automated feature selection

- **Model Training**
  - Multiple model support (Random Forest, Gradient Boosting, Logistic Regression)
  - Hyperparameter tuning
  - Ensemble methods
  - Model evaluation metrics

- **Model Explanation**
  - Feature importance analysis
  - SHAP values for tree-based models
  - LIME explanations
  - Model interpretability visualizations

- **Visualization**
  - Distribution plots
  - Correlation matrices
  - Scatter plots
  - Box plots
  - PCA visualizations
  - Feature importance plots

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/data_science_pipeline.git
cd data_science_pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure the pipeline by editing `config.yaml`:
```yaml
data:
  input_path: "data/raw/data.csv"
  target_column: "target"
  test_size: 0.2
  random_state: 42

# ... (other configurations)
```

2. Run the pipeline:
```bash
python pipeline.py
```

## Project Structure

```
data_science_pipeline/
├── data/
│   ├── raw/
│   └── processed/
├── preprocessing/
│   ├── data_cleaner.py
│   └── data_processor.py
├── feature_engineering/
│   └── feature_generator.py
├── models/
│   ├── model_trainer.py
│   └── model_explainer.py
├── visualization/
│   └── advanced_plots.py
├── results/
│   ├── metrics/
│   ├── models/
│   ├── plots/
│   └── explanations/
├── logs/
├── config.yaml
├── pipeline.py
├── requirements.txt
└── README.md
```

## Configuration

The pipeline is configured through `config.yaml`. Key configuration sections include:

- `data`: Input/output paths and data settings
- `preprocessing`: Data cleaning and preprocessing options
- `feature_engineering`: Feature generation settings
- `model`: Model training configuration
- `explanation`: Model explanation settings
- `visualization`: Plot generation options

## Output

The pipeline generates the following outputs:

1. **Processed Data**
   - Cleaned and preprocessed dataset
   - Feature-engineered dataset

2. **Model Results**
   - Trained models
   - Model metrics
   - Feature importance scores

3. **Visualizations**
   - Data distribution plots
   - Correlation matrices
   - Model performance plots
   - Feature importance plots

4. **Explanations**
   - SHAP values
   - LIME explanations
   - Feature importance analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 