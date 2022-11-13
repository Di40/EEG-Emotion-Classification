# EmotionRecognition
Emotion recognition from EEG data - [Graduation thesis](Diplomska_Dejan_Dichoski.pdf)

Emotion representation using Circumplex Model (arrousal and valence).

Dataset used: [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/)
32 people (16 male, 16 female), 40 trials per person, 32 electrodes, 128Hz


Feature extraction:
1. Time domain:
  - Statistical: mean of the raw signal over time N, standard deviation of the raw signal, mean of the absolute values of the first differences of the raw signal, mean of the absolute values of the first differences of the normalized signal, mean of the absolute values of the second differences of the raw signal, mean of the absolute values of the second differences of the normalized signal, skewness, kurtosis, 
  - Non-linear: mobility, complexity
2. Frequency domain (FFT):
  - Average and relative power (Welch method) for alpha, beta and theta bands
  - Multi‐Electrode Features: differential and rational assymetry (between left and right hemisphere)
3. Time-frequency domain (Wavelet transformation)
  - Energy and entropy

Train-test splitting (75 train : 12.5 validation: 12.5 test):
- Select all the trials for 4 people and keep them as a test set.
- For the rest of the people (28) and trials do cross-validation.
- Use GroupKFold (7 groups) because the dataset is ordered. As a result, all of the trials for a specific person are either in the training or the validation set. 1/7 is validation set i.e. all the trials for 4 people (40*4).

Classifiers: Gradient Boosting, AdaBoost, Random Forest, Extreme Gradient Boosting, Nearest Neighbors, Naive Bayes, SVM, Decision Tree.
Evaluation metrics: accuracy and F1.

Best classifier: SVM (F1: Valence 73.0, Arousal 67.01)
