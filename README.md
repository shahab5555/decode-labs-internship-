 AI Supervised Learning: Iris Data Classification

A Python-based machine learning project that demonstrates a complete **Input-Process-Output** pipeline for supervised learning. The project uses the **K-Nearest Neighbors (KNN)** algorithm to classify different species of Iris flowers based on their sepal features and visualizes the results.

Developed during my technical internship at **Decode Labs**.

 System Architecture

The codebase is organized into three clear, distinct phases:

1.  **INPUT PHASE:** Loads the classic Iris dataset and extracts the primary physical features (`Sepal Length` and `Sepal Width`) for 2D spatial analysis.
2.  **PROCESS PHASE:** 
    *   Splits the data into training ($80\%$) and testing ($20\%$) sets with shuffling enabled.
    *   Standardizes the features using `StandardScaler` to optimize spatial distance calculations.
    *   Trains a **K-Nearest Neighbors (KNN)** classifier with $K = 3$.
3.  **OUTPUT PHASE:** Evaluates the model using a comprehensive classification report (precision, recall, F1-score) and generates a standardized scatter plot using `seaborn`.

---

 Tech Stack & Dependencies

   **Language:** Python 3.x
   **IDE:** Visual Studio Code (VS Code)
   **Machine Learning:** `scikit-learn`
   **Data Visualization:** `matplotlib`, `seaborn`
   **Numerical Computing:** `numpy`

 📂 Project Structure
```text
├── main.py              # Core AI script (Input, Process, Output logic)
├── requirements.txt     # Python project dependencies
└── README.md            # Project documentation
