
# DeepDive-ANN-
This project uses an Artificial Neural Network (ANN) to detect fraudulent credit card transactions using a real-world dataset.

What is ANN?

An Artificial Neural Network (ANN) is a computer system designed to work like a human brain. It learns from examples instead of following strict rules.

Key Parts of an ANN

Neurons (Nodes) → Like brain cells that process information.

Layers → Input (data), Hidden (thinking), Output (answer).

Weights → How important each piece of data is.

Learning → Adjusts weights by seeing many examples (like practice).


1. Bias

A small number added to adjust the output (like a "cheat" to help the model fit better).

2. Weight

A number that decides how important an input is.

3. Forward Propagation
Passing data through the ANN to get an output (like a quiz guess).

Steps:

Input → Multiply by weights → Add bias → Apply activation function → Pass to next layer.

Repeat until output layer.

4. Backward Propagation (Backpropagation)

Fixing mistakes by adjusting weights/bias (like learning from wrong quiz answers).

Steps:

Compare ANN’s output to the correct answer (e.g., "100% cat").

Calculate error (how wrong it was).

Update weights/bias to reduce error (using math called gradient descent)

5 Activation Function

An activation function decides whether a neuron should "fire" (turn ON) or not based on its input.

Think of it like a light switch:

If the input is strong enough → Light ON (neuron activates).

If not → Light OFF (neuron stays quiet).

Types of Activation function

Step Function Definition: The simplest activation function that works like a light switch - it's either completely ON (1) or completely OFF (0) based on whether the input reaches a threshold.

Sigmoid (Logistic Function) Definition: A smooth S-shaped curve that squishes any input value to between 0 and 1, useful for giving probability-like outputs.

Tanh (Hyperbolic Tangent) Definition: Like a centered version of sigmoid that squishes inputs to between -1 and 1, making it better for some learning tasks.

ReLU (Rectified Linear Unit) Definition: The most popular activation that simply outputs the input if it's positive, or zero if negative - like a water faucet that only flows one way.

Leaky ReLU Definition: A smarter version of ReLU that never completely shuts off - it lets a tiny amount pass through even for negative inputs.

Softmax Definition: A special function that turns a bunch of numbers into probabilities that add up to 100%, perfect for picking between multiple options.

Swish Definition: A smooth, self-gated function discovered by Google that often works better than ReLU, especially in deep networks.

Linear (Identity Function) Definition: The simplest possible function that just outputs exactly what was input, with no changes or squishing.

ELU (Exponential Linear Unit) Definition: Like ReLU but uses exponential smoothing for negative inputs, helping solve some learning problems.

SELU (Scaled ELU) Definition: A self-normalizing version of ELU that automatically adjusts its output scale to help deep networks learn better.

Loss Functions

A loss function is like a report card for your neural network. It tells the model:

"How wrong were your answers?"

"How much should you improve?"

Types of Loss Functions

Mean Squared Error (MSE) Use: For predicting numbers (e.g., house prices, temperature).
How it works: Punishes big mistakes more (e.g., wrong by  
100
→
l
o
s
s
=
 10,000).

Binary Cross-Entropy Use: Yes/No problems (e.g., "Is this a cat?").
How it works: Harshly punishes wrong confidence (e.g., saying "99% cat" when it's a dog).

Categorical Cross-Entropy Use: Picking between multiple options (e.g., "Cat/Dog/Bird").
How it works: Checks if the model put high probability on the correct class. What Are Precision, Recall, and F1 Score? Imagine you're building a robot that detects cats in photos. Sometimes it gets it right, sometimes it makes mistakes. These three scores help you measure how good your robot is.

✅ Precision — How many of the "yes" answers were actually correct? Out of all the times the robot said “It’s a cat!”, how many were really cats?

📦 Formula (in Python): precision = true_positives / (true_positives + false_positives) Example: Robot said “Cat” 10 times, but only 7 were actually cats.

precision = 7 / 10 # = 0.7 or 70% So 70% of its “cat” answers were correct.

Recall — How many real cats did it find? Out of all the real cats, how many did the robot find and say "cat"?

📦 Formula (in Python): recall = true_positives / (true_positives + false_negatives) Example: There were 10 real cats in the photos. The robot only found 6 of them.

recall = 6 / 10 # = 0.6 or 60% So it found 60% of the actual cats.

F1 Score — The balance between precision and recall Simple Meaning: A score that combines precision and recall into one number. It's high only if both are high. f1_score = 2 (precision recall) / (precision + recall) Example: If precision = 0.8 and recall = 0.5

f1_score = 2 (0.8 0.5) / (0.8 + 0.5) = 0.615 or 61.5%

Overfitting & Underfitting

Overfitting → "Memorizing the Answers!" The model learns too many details from training data (including noise) and fails on new data.
Example:

A student memorizes all answers from a practice test but fails the real exam because questions are slightly different.

Signs:

99% accuracy on training data → 60% on test data.

Follows training data too closely (like fitting a squiggly line to simple data).

Fix:

Use more data (bigger practice tests).

Simplify the model (fewer neurons/layers).

Add regularization (like telling the model "don’t overcomplicate things!").

2. Underfitting → "Didn’t Study Enough!"

What? The model is too simple to learn patterns in the data.

Example:

A student only reads the chapter titles and guesses answers randomly.

Signs:

60% accuracy on training data → 55% on test data (bad everywhere).

The model’s predictions are too generic (like always saying "cat" for any animal).

Fix:

Use a more complex model (add layers/neurons).

Train longer (more "study time").

Add better features

✅ Techniques to Avoid Overfitting

Dropout — "Random Neuron Breaks" What it does: During training, Dropout randomly turns off some neurons so the network doesn’t rely too much on specific ones.

Simple Explanation: It forces the network to spread out learning and become more general.

import tensorflow as tf

Add dropout to a layer
tf.keras.layers.Dropout(rate=0.5) Example: Like asking different team members to take turns leading practice. Everyone improves, not just the star player.

⚖️ 2. Regularization — "Keeping It Simple" Regularization helps the model stay simple and not memorize too much. It adds a penalty to the weights during training.

🧩 L1 Regularization — "Make Some Weights Zero" What it does: It adds the absolute values of weights to the loss function.

This often results in sparse models, where some weights become exactly 0 — so the model focuses on fewer inputs.

📦 Formula: L1penalty = lambda * sum(abs(w)) 📦 Python Example: tf.keras.regularizers.l1(l=0.01) Analogy: Like cleaning a messy room — get rid of stuff you don’t need.

🧩 L2 Regularization — "Shrink the Weights" What it does: Adds the square of weights to the loss. This shrinks big weights, but rarely makes them exactly zero.

Helps the model stay smooth and less sensitive.



A brief description of what this project does and who it's for




![Logo](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/th5xamgrr6se0x5ro4g6.png)




## Roadmap
 Dataset

Source: Kaggle Credit Card Fraud Detection Dataset

Format: CSV inside a ZIP archive

Features: 30 columns (including anonymized PCA features, Time, Amount)

Target: Class

0 = Non-fraud

1 = Fraud

📌 Project Structure

creditcard.csv.zip – Raw dataset

ANN.py – Main training and evaluation script

model_results/ – Output folder with saved graphs and evaluation metrics

model_results.txt – Metrics and model summary

train_conf_matrix.png – Confusion matrix on training set

val_conf_matrix.png – Confusion matrix on validation set

metrics_bar.png – Accuracy, Precision, Recall, F1 Score

val_prediction_distribution.png – Class distribution of predicted values

⚙ How the Code Works
1. Data Loading

Extracts the zip file and reads the CSV using pandas

2. Preprocessing

Splits the dataset into training and validation sets (80/20 split)

Scales the features using StandardScaler

3. Model Architecture

A simple 3-layer Sequential ANN:

Dense(32, relu)

Dense(16, relu)

Dense(1, sigmoid)

Loss: binary_crossentropy

Optimizer: adam

4. Training

Trained for 10 epochs with batch size 2048

5. Evaluation

Predicts on both training and validation sets

Computes:

Accuracy

Precision

Recall

F1 Score

Confusion Matrices

6. Visualization

Generates and saves the following plots:

Confusion matrices (training & validation)

Bar chart for metrics

Predicted class distribution chart

7. Export Results

Saves model summary, metrics, and confusion matrices into model_results/model_results.txt

 Example Visualizations

Confusion Matrix (Validation)
Image saved: model_results/val_conf_matrix.png

Model Metrics

Image saved: model_results/metrics_bar.png

Predicted Class Distribution

Image saved: model_results/val_prediction_distribution.png

🧪 Performance Metrics (Sample Output)

TRAINING METRICS:
Accuracy: 0.9994
Precision: 0.86
Recall: 0.81
F1 Score: 0.83

VALIDATION METRICS:

Accuracy: 0.9992
Precision: 0.89
Recall: 0.76
F1 Score: 0.82

Note: Actual results may vary depending on random seed and dataset split.



