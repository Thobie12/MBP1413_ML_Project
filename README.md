# MBP1413H – Biomedical Applications of Artificial Intelligence  
## Assignment Project (Jupyter Notebook)

### Overview
This repository contains a Jupyter Notebook developed as part of the **MBP1413H: Biomedical Applications of Artificial Intelligence** course.

In this project, we implemented a complete machine learning pipeline to analyze biomedical data and build predictive models. The workflow includes:

- Data preprocessing and cleaning
- Feature engineering and selection
- Handling class imbalance
- Model training using multiple algorithms (e.g., XGBoost, Neural Networks)
- Hyperparameter tuning
- Model evaluation using appropriate performance metrics
- Model interpretation (e.g., SHAP analysis)

The notebook demonstrates how AI techniques can be applied to biomedical datasets to generate clinically relevant insights and predictive models.

---

### Repository Structure

├── AI_project_Uditha_Tobi_apr2.ipynb  # Main Jupyter Notebook  
├── README.md                          # Project documentation  
├── environment.yml                    # Conda environment file  
├── data/ Input                        # Input dataset  
└── data/ Output                       # Output folder

---

### Environment Setup

This project uses:

- Python 3.10.20  
- Conda (Miniforge/Anaconda recommended)

---

### Create Environment from YAML

```bash
conda env create -f environment.yml
conda activate mbp1413h_env
```

---

### Running the Notebook

1. Activate the environment:
   ```bash
   conda activate mbp1413h_env
   ```

2. Launch Jupyter:
   ```bash
   jupyter lab
   ```
   or
   ```bash
   jupyter notebook
   ```

3. Open:
   AI_project_Final.ipynb

4. Run all cells sequentially.

---

### Notes

- CPU execution is fully supported.
- Apple Silicon users benefit from tensorflow-macos and tensorflow-metal.
- Ensure dataset paths are configured correctly before execution.

---

### Key Libraries

- scikit-learn  
- xgboost  
- tensorflow / keras  
- pandas / numpy  
- shap  
- statsmodels  

---

### Author
Uditha Maduranga & Oluwatobi Agbede

---

### 📄 License
Academic use only.
