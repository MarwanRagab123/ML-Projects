# Customer Segmentation using K-means Clustering

## Project Overview
This project implements customer segmentation using the K-means clustering algorithm on the Mall Customers dataset. The goal is to identify distinct customer groups based on their spending patterns and demographic information.

## Features
- Customer segmentation analysis
- K-means clustering implementation
- Data visualization
- Cluster analysis and interpretation
- Performance metrics evaluation

## Dataset
The project uses the Mall Customers dataset which includes:
- Customer ID
- Gender
- Age
- Annual Income
- Spending Score

## Technical Details

### Implementation
- K-means clustering algorithm
- Optimal cluster number determination using:
  - Elbow method
  - Silhouette analysis
- Feature scaling and preprocessing
- Cluster visualization using scatter plots

### Tools Used
- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure
- `Kmeans.ipynb`: Main Jupyter notebook containing the analysis
- `kmean.py`: Python script with K-means implementation
- `Mall_Customers.csv`: Dataset file

## Requirements
```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

## Installation
1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the Jupyter notebook:
```bash
jupyter notebook Kmeans.ipynb
```

## Results
- Identified distinct customer segments
- Visualized cluster patterns
- Analyzed customer characteristics for each segment
- Provided insights for targeted marketing strategies

## Future Improvements
1. Implement other clustering algorithms
2. Add more features for segmentation
3. Real-time customer segmentation
4. Integration with marketing automation tools
5. Dynamic cluster number optimization

## Author
[Marwan Ragab]

