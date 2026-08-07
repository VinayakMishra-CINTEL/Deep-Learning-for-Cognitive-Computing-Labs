# Lab 1: Backpropagation ANN – Observations & Analysis

## Part B – Observation & Analysis

### Q1. What is the purpose of the forward propagation step?
Forward propagation is how the network turns input sensor values into a prediction.
Each input is multiplied by its weights, summed with a bias, and passed through an
activation function — first for the hidden layer, then for the output layer. It
produces the network's current "guess" (a2), which is then compared against the
actual label to calculate the error.

### Q2. Why is the Sigmoid activation function used in this experiment?
Sigmoid squashes any input into a range between 0 and 1, which is ideal for binary
classification (Acceptable/Defective) since the output can be directly interpreted
as a probability. It is also smooth and differentiable everywhere, which is required
for backpropagation to calculate gradients.

### Q3. Record the initial and final loss values. What do these values indicate?
- **Initial MSE:** 0.2446
- **Final MSE (5000 epochs):** 0.0695

The large drop (about 72% reduction) shows the network successfully learned the
relationship between sensor readings and product quality. A high initial loss
reflects random, untrained weights; the much lower final loss shows the weights
converged toward values that fit the training data well.

### Q4. Why is Backpropagation considered the learning mechanism of an ANN?
Backpropagation is what allows the network to actually improve. It calculates how
much each weight contributed to the final error (using the chain rule) and pushes
that error signal backward from the output layer to the hidden layer. Without it,
the network would have no way to know *which* weights to adjust or by *how much* —
it's the mechanism that converts a measured error into concrete weight updates.

### Q5. What happens if the learning rate is increased from 0.1 to 1.0?
Observed result (2000 epochs):

| Learning Rate | Final MSE | Test Accuracy |
|---|---|---|
| 0.1 | 0.0713 | 90.0% |
| 1.0 | 0.5062 | 48.0% |

At lr=1.0, training became unstable — the weight updates overshot the error
minimum on every step, causing the loss to stay high (worse than the untrained
starting error) and accuracy to collapse to near-random (48%, since classes are
~50/50). This shows a learning rate that's too large prevents convergence rather
than speeding it up.

### Q6. Epochs vs Final Error vs Accuracy

| Epochs | Final Error | Prediction Accuracy |
|---|---|---|
| 100 | 0.1088 | 88.00% |
| 1000 | 0.0728 | 89.50% |
| 5000 | 0.0695 | 89.00% |

**Comment:** Most of the learning happens early — error drops sharply between 100
and 1000 epochs. Beyond that, additional epochs give diminishing returns: the error
barely changes from 1000 to 5000 epochs, and accuracy doesn't keep improving (it
even dips slightly at 5000). This shows the network converges to a near-optimal fit
well before 5000 epochs, and further training mostly fine-tunes without meaningfully
improving performance.

### Q7. If hidden layer has 10 neurons instead of 4, will it always perform better?
Observed result (5000 epochs):

| Hidden Neurons | Final MSE | Test Accuracy |
|---|---|---|
| 4 | 0.0695 | 89.00% |
| 10 | 0.0665 | 89.50% |

No, more neurons do not always mean better performance. The improvement here is
marginal (+0.5% accuracy) because the underlying pattern in the data isn't very
complex — 4 neurons were already enough to capture it. Adding more neurons increases
model capacity and computation cost, but beyond a certain point gives negligible
benefit, and on harder or smaller datasets it can even cause overfitting
(memorizing training data instead of generalizing).

### Q8. Why are weights initialized randomly instead of all zeros?
If all weights were initialized to zero, every neuron in a layer would receive the
exact same input and compute the exact same gradient during backpropagation. They
would all update identically forever, meaning the hidden layer would behave like a
single neuron no matter how many neurons it actually has — this is called the
**symmetry problem**. Random initialization breaks this symmetry so each neuron
learns a different feature.

### Q9. Role of the learning rate in Backpropagation
The learning rate controls the size of each weight-update step during gradient
descent.
- **Too small:** Learning becomes very slow — the network needs many more epochs
  to converge, and training may appear to "stall" even though it's still improving.
- **Too large:** As shown in Q5, updates overshoot the error minimum, causing the
  loss to oscillate or diverge instead of settling down — the network may fail to
  learn at all (observed: accuracy collapsed to 48% at lr=1.0).

The learning rate must be tuned to balance convergence speed against stability.

### Q10. Three real-world applications of Backpropagation-based neural networks
1. **Image recognition** – e.g. classifying medical scans, facial recognition.
2. **Speech recognition** – converting spoken audio into text (voice assistants).
3. **Fraud detection** – classifying financial transactions as legitimate or
   fraudulent based on transaction patterns (directly analogous to this lab's
   Acceptable/Defective classification).

---

## Critical Thinking: Sigmoid vs ReLU in the hidden layer

| | Sigmoid (lr=0.1) | ReLU (lr=0.1) | ReLU (lr=0.01) |
|---|---|---|---|
| Final MSE | 0.0695 | 0.5062 (diverged) | 0.0697 |
| Test Accuracy | 89.0% | 48.0% (diverged) | 89.5% |
| Convergence | ~1000 epochs | never converged | ~500–1000 epochs |

**Discussion:**
Using the same learning rate (0.1) that worked for sigmoid caused ReLU to diverge —
MSE jumped to 0.506 and stayed flat, with accuracy collapsing to 48% (random
guessing). This happens because ReLU has no upper bound (unlike sigmoid, which is
always capped between 0 and 1), so its gradients can be much larger, causing weight
updates to explode.

Once the learning rate was reduced to 0.01, ReLU trained successfully and reached
almost identical performance to sigmoid (MSE 0.0697 vs 0.0695, accuracy 89.5% vs
89.0%), converging in a similar or slightly fewer number of epochs.

**Conclusion:** For this problem, ReLU does not meaningfully improve performance
over sigmoid — both reach essentially the same final loss and accuracy. However,
the experiment reveals an important practical lesson: **ReLU is far more sensitive
to learning rate** than sigmoid, and requires careful tuning to avoid instability,
even though it is generally preferred in deeper networks for avoiding vanishing
gradients.
