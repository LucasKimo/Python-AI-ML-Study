- **Machine Learning:**  Learns patterns from data to make predictions or decisions
- **Deep learning:** A subset of machine learning that uses multi-layer neural networks to learn more complex patterns. 
- Machine Learning is often used for structured data and simpler prediction tasks
- Deep Learning is better for images, speech, and text.

# Main Types of Machine Learning
### Supervised Learning
- uses labelled data
- learns from input data and the correct answers
### Unsupervised Learning
- finds hidden structure without labels
- finds patterns or clusters in data
### Reinforcement Learning
- learns through rewards and penalties
- learns by trial and error through feedback

# Common Machine Learning Models
| Model               | Role                                                  | Common use                                                                                                                                                                                                                                                                           |
| ------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Linear regression   | Predicts **continuous values**                        | House prices, sales, temperature [cloud.google](https://cloud.google.com/learn/what-is-machine-learning?hl=ko)                                                                                                                                                                       |
| Logistic regression | Predicts **class probability**                        | Spam detection, pass/fail classification linkedin+1                                                                                                                                                                                                                                  |
| Decision tree       | Uses **rule-based splits for prediction**             | Problems that need interpretability [databricks](https://www.databricks.com/kr/blog/what-are-machine-learning-models)                                                                                                                                                                |
| Random forest       | Combines many decision trees for stronger performance | General classification and regression tasks [s1275702.tistory](https://s1275702.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8%EC%9D%98-%EC%A2%85%EB%A5%98%EC%99%80-%EA%B5%AC%ED%98%84-%EB%B0%A9%EB%B2%95-%EC%99%84%EB%B2%BD-%EA%B0%80%EC%9D%B4%EB%93%9C) |
| SVM                 | Separates classes with an optimal boundary            | Classification problems with clear margins [linkedin](https://kr.linkedin.com/pulse/types-machine-learning-models-from-basics-advanced-naresh-maddela-wsqhc?tl=ko)                                                                                                                   |
| KNN                 | Classifies based on nearby examples                   | Small and simple datasets [linkedin](https://kr.linkedin.com/pulse/types-machine-learning-models-from-basics-advanced-naresh-maddela-wsqhc?tl=ko)                                                                                                                                    |
| Naive Bayes         | Classifies using probability                          | Text classification, spam filtering [linkedin](https://kr.linkedin.com/pulse/types-machine-learning-models-from-basics-advanced-naresh-maddela-wsqhc?tl=ko)                                                                                                                          |
| K-means             | Groups similar data points                            | Customer segmentation, clustering [databricks](https://www.databricks.com/kr/blog/what-are-machine-learning-models)                                                                                                                                                                  |
| PCA                 | Reduces dimensions while keeping key information      | Visualization, preprocessing, noise reduction [databricks](https://www.databricks.com/kr/blog/what-are-machine-learning-models)                                                                                                                                                      |
## When to use each model
### Number Prediction
- linear regression
- random forest regression
### Category Prediction
- logistic regression
- SVM
- decision tree
### Grouping similar items
- K-means
### Reducing many features
- PCA
### Text Classification
- naive Bayes


### Linear Regression
- Predicts continuous number
- Tries to find the best straight-line relationship between input variables and the output
- Use it when the answer should be value, such as house price, salary, or temperature. 
### Logistic Regression
- Used for classification
- Estimates the probability that something belongs to a certain class, such as spam vs not spam or yes vs no
- Use it when you want a clear, efficient model for **binary classification.**
### Decision Tree
- Makes decisions by asking a series of questions, like a flow chart
- Each split divides the data **into smaller groups** until the model reaches a prediction
- Use it when you want something easy to understand and explain.
- Ranks which variables had the biggest impact on the final outcome.
- Discovering hidden patterns
- **Intuitive**, but a **single tree** can **overfit** if it becomes too **complex**'
### Random Forest
- Collection of many decision trees
- Each tree gives its own prediction, and the forest combines them for a more stable final result.
- Use it when you want better accuracy and less overfitting than a single decision tree
- Works well on tabular data
- Reliable general-purpose model
### SVM (Support Vector Machine)

