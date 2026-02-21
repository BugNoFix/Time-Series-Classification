
## 1. Introduction

The goal of this project is to achieve time series classification with a deep learning model. For this purpose, we have incrementally developed several models and used different approaches and techniques.

The report is structured as follows:
* **Problem Analysis**: We explain which kind of data exploration we did and how and why we preprocess the data in a certain way.
* **Models and Methods**: We explain the ideas we had, the models we implemented, and why we thought they could be useful.
* **Experiments**: We show the metrics obtained by the average best runs of relevant models.
* **Results**: We explain the results gotten in light of the problem structure.
* **Discussion**: We analyze critically our results.
* **Conclusion**: We give information on possible future improvements.

---

## 2. Problem Analysis

### Dataset Exploration
> *Code can be found in `DataExploration.ipynb`*

Initially, an exploratory analysis of the dataset was conducted to understand its characteristics.
The class distribution was examined, showing a clear imbalance: **77.3%** of the samples belonged to the *no pain* class, **14.2%** to *low pain*, and **8.5%** to *high pain*. Such imbalance required appropriate handling to ensure adequate model performance; for instance, we employed class weights to adjust the cross-entropy loss to penalize errors on classes with fewer samples more heavily.

One-hot encoding was applied to the body-related features, and a correlation matrix was computed for all variables. The features `n_eyes`, `n_legs`, and `n_hands` were found to be perfectly correlated (correlation coefficient equal to 1), meaning they carried redundant information; consequently, two of them were removed. Additionally, the feature `joint_30` was constant across all samples, therefore non-informative, and was discarded.

It was also verified that each user sequence consisted of exactly 160 timestamps, confirming the absence of missing segments in the time series. Outlier detection was performed at the row level using interquartile ranges, but due to the skewed nature of many feature distributions, a large number of rows would have been flagged as outliers. To avoid losing potentially relevant information, no rows were removed.

### Dataset Pre Processing
> *Code can be found in `DataPreProcessing.ipynb`*

The continuous features were subsequently normalized, and the labels were mapped into numerical values. In order to build windows to feed to our models, we first grouped the rows by sample index (to make the code more robust and future-proof, we also added padding logic even though each user already had exactly 160 samples), and then we created functions to build windows given the width and the stride.

Next, the autocorrelation was computed for each feature to determine the appropriate window and stride sizes; however, the results did not reveal any meaningful patterns. Consequently, several configurations were tested, and it was empirically determined that the best-performing values were a **window size of 32** and a **stride size of 16**.

The choice of the train–validation split was also based on an empirical approach. Four different splits were evaluated: 80%–20%, 85%–15%, 90%–10%, and 95%–5%. The **90%–10% split** yielded the best performance. When smaller training splits were used, the model showed reduced generalization capabilities, likely because the training set became too small compared to the test set on Kaggle, making it necessary to use as much data as possible for training.

Accordingly to the competition overview guidelines, the **F1 score** was employed for model evaluation.

---

## 3. Models and Methods

### Basic Architectures
> *Code can be found in `rnn.ipynb`, `egru.ipynb` and `cnngru.ipynb`*

The initial phase of the study involved the implementation of classical deep learning architectures for sequential data, namely **RNN**, **LSTM**, and **GRU**.
These models were used to establish baseline performance levels and to assess the ability of recurrent structures to capture temporal patterns in the dataset.

A hybrid **CNN+GRU** model was also evaluated, in which convolutional layers extract short-range temporal features before passing them to a GRU classifier. However, it was soon discarded because the improvement in performance was not impressive and, above all, the validation loss curve and the validation F1 score curve were too spiky; even fine-tuning the hyperparameters did not help significantly.

We made a discrete jump forward when we inserted into our best stable model (the GRU) the embeddings of the columns regarding the pain surveys, which before were fed along with continuous features, thus creating the so-called **EGRU** model. To manage the data smoothly, we created a structured feature dictionary enabling a clear separation and processing of the different feature categories.

### Advanced Architectures
> *Code can be found in `transformer.ipynb`*

The EGRU was finally achieving good scores, but to improve further, we needed more sophisticated models able to better capture the temporal dynamics of the data.
A **Transformer** architecture was employed to leverage self-attention mechanisms capable of modeling global dependencies across the input sequence.

At this stage, since the model was much larger, we tried to shrink down the number of continuous features and therefore investigated the use of an **autoencoder** to obtain a compact latent representation of the input features prior to classification. However, this dimensionality reduction led to the loss of relevant information, ultimately resulting in reduced classification performance.

### Ensemble Approach
> *Code can be found in `ensemble.ipynb`*

To further enhance classification performance, we decided to create a completely different third model: a **Temporal Convolutional Network (TCN)**. This is a neural network for sequence data that uses 1D causal convolutions; compared to RNNs, it processes sequences in parallel, is easier to train, and is often more stable.

We built an **ensemble model** by combining its predictions with those of the **EGRU** and the **Transformer**. We chose to use only three models to minimize the number of ties. Moreover, this ensemble exploits the complementary strengths of the three architectures: temporal convolutional extraction, recurrent memory, and global self-attention, resulting in more robust and stable performance across experiments. The idea is that each of these models has an architecture that leverages different features and correlations among windows, thus giving a more general perspective on the problem.

---

## 4. Experiments

The F1-score reported in the table below is calculated on the validation set.
We report the average F1 score on the public test set for each model (the average refers to the model tuned with the best or comparable-to-best hyperparameters choice).

### Avg model performance on public test set

| Model | F1-score (weighted) |
| :--- | :---: |
| GRU | 84.31 |
| EGRU | 91.78 |
| Transformer | 92.71 |
| **Ensemble** | **95.67** |

---

## 5. Results

The results we obtained are satisfying. The ensemble model seems to perform well and is flexible enough to reach a really good F1 score. It is much better than a standard GRU model (our initial baseline), and the use of three different models improves the overall performance also with respect to a standalone Transformer architecture, which is already an advanced and powerful architecture.

---

## 6. Discussion

In this section, the results are analyzed critically, considering strengths, weaknesses, limitations, and assumptions.
The ensemble model demonstrated strong performance by combining the complementary capabilities of TCN, EGRU, and Transformer architectures.

However, certain weaknesses were observed. In particular, the model is sensitive to tie situations, which occur in two cases:

* **Window-level tie:** When each model predicts a different class for the same input window, the final prediction for that window is chosen randomly with the same probability for each of the three classes.
* **User-level tie:** When, after aggregating predictions from all windows of a user, no class reaches a majority, the final user-level prediction is assigned randomly with the same probability for each of the three classes.

These situations highlight that the ensemble is less reliable when the models strongly disagree, but that rarely happens.

Moreover, the ensemble model is heavier to train since we are training three separate networks, and it is more difficult to fine-tune because we have to find the right tradeoff of hyperparameters not only for each model but also with respect to one another.

---

## 7. Conclusions

A potential future direction might be creating an ensemble composed of multiple instances of the same or different models, each trained using different window sizes.
This approach would allow each instance to capture slightly different temporal patterns, providing diverse perspectives on the input sequences.
