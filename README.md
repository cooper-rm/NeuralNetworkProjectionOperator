# Neural Network Projection Operator

Investigating whether direct geometric manipulation of neural network activations can improve learned representations — starting with centroid projection on MNIST.

---

# Part 1: Project Selection & Setup

**Project Pathway:** 
    -   Research Investigation

**Pathway justification** 

    -   Research Investigation fits my project since it directly operates on neural netowrks in uncommon or novel ways. 

**Dataset:** MNIST (Modified National Institute of Standards and Technology)

**Source:** `torchvision.datasets.MNIST` (originally from [Yann LeCun's MNIST page](http://yann.lecun.com/exdb/mnist/))

**Dataset Description** 
    -   The dataset I have selected for use is the MNST Image Dataset. This dataset contains 60,000 training mages and 10,000 testing images of handwritten digits. I selected this dataset since it is simple and thus I will be able to veryify my findings Deep learning is appropriate since the novelty of this research focuses directly on neural networks and I will be using classification accuracy to measure success.   

### Problem Statement 

A common problem with neural networks when working with noisy datasets or complex datasets is the inability to meaningfully learn the difference between certain class items. Neural networks achieve high accuracy when they can learn to map inputs to meaningful output representaions. This project consists of novel research focused on answering the question: Does spacing out clustering in the neural network layer outputs as an embedding space improve representations and increase classification accuracy? I chose to perform research into neural networks and image embedding spaces because my current understanding is that embedding spaces are a reflection of neural networks. My assumption is that a more spaced out embedding space leads to higher classification accuracy. The dataset I have selected for use is the MNST Image Dataset. This dataset contains 60,000 training mages and 10,000 testing images of handwritten digits. I selected this dataset since it is simple and thus I will be able to veryify my findings Deep learning is appropriate since the novelty of this research focuses directly on neural networks and I will be using classification accuracy to measure success.   

**Why deep learning is the right tool for this problem:**

The core purpose of neural networks is to learn how to map inputs to meaningful outputs This process involves representational geometry which is something that exists primarily in deep learning methods. Deep learning isn't the right tool its the core domain this research focuses on. 

**Evaluation metric and justification:**

    -   Classification accuracy: the practical bottom line — does the intervention help the model classify?

    -   Silhouette score: measures whether activations cluster tightly by class. Higher = points closer to their own centroid and farther from others.

    -   Logistic Regression classification Accuracy on hidden-layer activations: measures whether a linear classifier can separate classes in the learned representation. 


**Personal connection or interest:**

    -   My personal interest and previous research lies in embedding space geometry, so working through novel methods helps me move the field forward. 

---

# Part 2: Project EDA

### EDA Summary

**Key findings from your EDA:**

1. Near-balanced classes: per-class training counts range from 5,421 (digit 5) to 6,742 (digit 1), a max/min ratio of ~1.24. No resampling or weighted loss needed.

2. High sparsity and low intrinsic dimensionality: ~81% of all pixels are exactly zero (black background). PCA shows only 87 components capture 95% of the total variance out of 784 pixels, confirming the data lives in a low-rank subspace and the 784→128 first layer compresses heavily redundant input.

3. Clear cluster structure with specific overlap zones: both t-SNE and UMAP projections reveal well-separated digit clusters, but with consistent overlap between visually similar pairs (4/9, 3/5, 7/9). Per-class standard deviation images show the highest variation occurs in stroke geometry (loops, crossbars, angles) rather than stroke centers, explaining why these pairs confuse models.

**Potential challenges identified:**

1. Digit pairs 4/9, 3/5, and 7/9 share similar stroke geometry, creating natural overlap in feature space. Any representation learning method must resolve these specific confusions to improve accuracy.

2. Outlier detection via distance-to-centroid revealed genuinely atypical writing styles and potential labeling ambiguities in the training set, meaning some misclassifications may reflect label noise rather than model failure.

**Preprocessing decisions based on EDA:**

1. Standard normalization (mean=0.1307, std=0.3081) applied to all inputs. No cropping or augmentation since the EDA confirmed digits are already centered and the sparsity pattern is consistent across classes.

2. No class rebalancing applied — the near-uniform distribution means standard cross-entropy loss is appropriate without weighting.

---

# Part 3: Model Development

### Experiment 01 — Regularization Ablation (784→128→64→10)

Each notebook adds one technique cumulatively. All use the same architecture, seeds, and 5-epoch training.

| NB | Technique | Cumulative Stack | Test Acc | Silhouette (64D) |
|----|-----------|------------------|----------|------------------|
| 01 | Baseline MLP | — | 0.9639 | 0.2748 |
| 02 | + Weight Decay (1e-4) | L2 | 0.9648 | 0.2763 |
| 03 | + BatchNorm | L2 + BN | 0.9727 | 0.4621 |
| 04 | + Dropout (0.3) | L2 + BN + Dropout | 0.9677 | 0.5189 |
| 05 | + L1 Regularization (1e-4) | L2 + BN + Dropout + L1 | 0.9691 | 0.5059 |
| 06 | + LayerNorm on h2 | L2 + BN(h1) + LN(h2) + Dropout + L1 | 0.9662 | 0.5379 |

### Experiment 01 — Centroid Projection Variants

Built on the full regularization stack (NB06). All use `model.fc3` as classifier.

| NB | Technique | Test Acc | Silhouette (64D) |
|----|-----------|----------|------------------|
| 07 | + Centroid Projection (Last Layer) | 0.9578 | 0.5041 |
| 08 | + Centroid Projection (All Layers) | 0.9000 | 0.4049 |
| 09 | + Centroid Inference (Last Layer) | 0.9553 | 0.6213 |
| 10 | + Centroid Inference (All Layers) | 0.6768 | 0.1731 |

### Experiment 02 — Clustering Pre-Training

Two-phase training: Phase 1 trains a backbone with a clustering objective, Phase 2 freezes the backbone and trains a classifier head. NB03-06 extend the simplex backbone with additional trainable layers (fc3→fc4→fc5).

| NB | Technique | Architecture | Test Acc | Silhouette |
|----|-----------|-------------|----------|------------|
| 01 | Cluster→Classify | 784→128→64→10 | 0.7054 | 0.4064 (64D) |
| 02 | Simplex→Classify | 784→128→64→10 | 0.9654 | 0.7623 (64D) |
| 03 | Extended Simplex | +fc3(48)+fc4(32)→10 | 0.9626 | 0.7506 (32D) |
| 04 | Extended (Low Reg) | +fc3(48)+fc4(32)→10 | 0.9652 | 0.7993 (32D) |
| 05 | Extended (No Reg) | +fc3(48)+fc4(32)→10 | 0.9654 | 0.7184 (32D) |
| 06 | From-Scratch Control | +fc3(48)+fc4(32)→10 | 0.9785 | 0.7884 (32D) |

---

# Part 4: Experimental Tracking

### Experiment 1: Baseline MLP (NB01)

**Architecture:** 784→128→64→10, ReLU activations, no regularization

**Hyperparameters:** Adam lr=0.001, batch_size=512, 5 epochs

**Hypothesis:** Without regularizations I expect the neural network to overfit. 

**LLM input:** Create a plain neural network with 3 layers as our research baseline. this should be small and go 128 to 64 to 10. 

**Results:**
- Training loss: 0.1100 (epoch 4)
- Validation: 0.9598 acc
- Test: 0.9639 acc, Silhouette: 0.2748

**Observations and conclusions:** The baseline achieves ~96% accuracy but the silhouette score of 0.27, which wasnt overfitting but confirms that the 64D hidden representations have weak cluster structure. The classes are overlapping and do not clearly group together. 

**Next steps:** The next step is to add standard regularization techniques one at a time to see which ones improve both accuracy and cluster quality.


### Experiment 2: Regularization Ablation (NB02–06)

**Architecture:** Same 784→128→64→10, adding one technique per notebook cumulatively from NB01 to NB06

**Hyperparameters:** Same base (Adam lr=0.001, batch_size=512, 5 epochs) + weight_decay=1e-4, dropout=0.3, l1_lambda=1e-4

**Hypothesis:** Adding regularization techniques should improve the model accuracy and cluster, and further prevent overfitting. 

**LLM input:** Add regularization techniques in a layered fashioned creating one new notebook at a time. these should follow the baseline structure, but include weight decay, batch norm, dropout, layer norm and any other techniques you deem valuable. 

**Results:**
- NB02 (+L2): test 0.9648, sil 0.2763 — minimal change from baseline
- NB03 (+BN): test 0.9727, sil 0.4621 — biggest accuracy jump, silhouette nearly doubled
- NB04 (+Dropout): test 0.9677, sil 0.5189 — accuracy dropped slightly but silhouette improved
- NB05 (+L1): test 0.9691, sil 0.5059 — accuracy improved over dropout, silhouette slightly decreased
- NB06 (+LN): test 0.9662, sil 0.5379 — highest silhouette in the ablation

**Observations and conclusions:** Weight decay alone (NB02) barely changed cluster structure (sil 0.27→0.28). BatchNorm was the single biggest improvement boosting accuracy (+0.9pp) and substantially improving sillouhette score (sil 0.28→0.46). Interestingly, dropout hurt accuracy slightly but continued improving silhouette, suggesting it forces more separable embedding representations. The best cluster structure came from the full stack with LayerNorm (sil 0.54), even though pure BatchNorm had the best accuracy.

**Next steps:** The next step is to add the centroid projection function and measure the results.

### Experiment 3: Centroid Projection (NB07–10)

**Architecture:** Same as NB06 + CentroidProjection module applied to hidden activations. Centroids recomputed each epoch from training data and activations are nudged toward their own class centroids and away from others.

**Hyperparameters:** proj_alpha=0.3, proj_beta=0.1. NB07/09 project last hidden layer only. NB08/10 project all hidden layers. NB09/10 also enable projection at inference time using nearest-centroid pseudo-labels.

**Hypothesis:** By projecting embeddings closer to their true centroid and away from others we should be able to create a more seperable embedding space and imporve accuracy. 

**LLM input:** Implement a centroid projection function as norebook 07 08 09 and 10 that nudges hidden activations toward their class centroid during the forward pass. Test it on the last layer only, then on all layers, and add an inference-mode variant that assigns pseudo-labels via nearest centroid. Make sure to add the centroid projection operator to the regularized baseline. Create 4 variants as the 4 different notebooks: project last layer only, project all layers, and both with inference-time projection enabled.

**Results:**
- NB07 (+Proj Last): test 0.9578, sil 0.5041 — accuracy dropped, silhouette dropped slightly vs NB06
- NB08 (+Proj All): test 0.9000, sil 0.4049 — significant accuracy degradation
- NB09 (+Infer Last): test 0.9553, sil 0.6213 — best silhouette in all of exp01
- NB10 (+Infer All): test 0.6768, sil 0.1731 — collapsed

**Observations and conclusions:** All versions reduced the testing accuracy substantially, however inference-mode projection on the last layer (NB09) achieved the highest silhouette of any exp01 model (0.62). 

**Next steps:** These results are showing no value in actual application ver standard approaches. Since we are not working on gradients we need to ry a fundamentally different approach. SO, instead of nudging activations inbetween layers with no back propigation, we can ftrain the backbone with an explicit clustering objective first, then classify on the frozen features. 

### Experiment 4: Clustering Pre-Training (Exp02 NB01–02)

**Architecture:** Same 784→128→64 backbone. Phase 1: train with clustering loss (intra-class pull + inter-class push). Phase 2: freeze backbone, add 64→10 head, train with CrossEntropyLoss. NB01 uses centroid-distance clustering loss; NB02 uses simplex targets (fixed equidistant target vectors).

**Hyperparameters:** Phase 1: margin=10.0, intra_weight=1.0, inter_weight=0.5, 5 epochs. Phase 2: Adam lr=0.001, 5 epochs, only 650 trainable params (fc3).

**Hypothesis:** Since we in theory want cleanly sepaerated clusters training a backbone on seperation and then adding and retrianing may improve model accuracy and silhoutte score. 

**LLM input:** Design a two-phase training approach as the next notebooks. First train the backbone using only a clustering loss that pulls the same class activations together and pushes different-class centroids apart, then freeze the backbone and train a linear classifier on top. Also try a simplex variant where the clustering targets are fixed equidistant vectors instead of learned centroids.

**Results:**
- NB01 (Cluster→Classify): test 0.7054, sil 0.4064 — poor classification but decent clustering
- NB02 (Simplex→Classify): test 0.9654, sil 0.7623 — strong accuracy AND best silhouette of any 64D model

**Observations and conclusions:** Raw cluster pre-training achieved better cluster structure (sil 0.41) but the representations weren't discriminative enough for a linear classifier at only 70.5% accuracy. The simplex approach was a breakthrough as it it matched the best regularized baseline in accuracy (96.5%) while achieving dramatically better cluster quality (sil 0.76 vs 0.54). This is the strongest model so far showing that embedding geometry and classification accuracy are related. The simplex pre-training optimized for geometry explicitly and got better results in both accuracy and geometry. With that said this model was undefitting heavily. This suggests that improved clustering may result in better inference time generalization

**Next steps:** Next we should attempt to reduce underfitting by adjusting regularization.

---

### Experiment 5: Extended Architectures (Exp02 NB03–06)

**Architecture:** Frozen simplex backbone (784→128→64) + new trainable layers fc3(64→48) + fc4(48→32) + fc5(32→10). NB06 trains the same extended architecture from scratch (no pre-training) as a control.

**Hyperparameters:** NB03: dropout=0.3, l1=1e-4. NB04: dropout=0.1, l1=1e-5 (low reg). NB05: no dropout, no L1 (no reg). NB06: same as NB04 but random init. All use ReduceLROnPlateau scheduler.

**Hypothesis:** Additional layers on frozen simplex features should learn more complex decision boundaries. SInce the clustering is substantially improved but underfitting we need more room for learning. 

**LLM input:** Take the simplex backbone and extend it with two additional trainable layers (64→48→32→10) while keeping the backbone frozen. Test with standard regularization, low regularization, and no regularization. Also train the same extended architecture from scratch as a control to see if the simplex pre-training actually helps.

**Results:**
- NB03 (Extended, standard reg): test 0.9626, sil 0.7506 (32D)
- NB04 (Extended, low reg): test 0.9652, sil 0.7993 (32D)
- NB05 (Extended, no reg): test 0.9654, sil 0.7184 (32D)
- NB06 (From-scratch control): test 0.9785, sil 0.7884 (32D)

**Observations and conclusions:** The from-scratch control outperformed all frozen-backbone variants, reaching 97.85% test accuracy. This is the highest of any model in the project. This suggests that freezing the backbone limited the extended models as the simplex features were good but not optimal for deeper architectures. The 32D silhouette scores (0.72–0.80) are high across all variants, indicating the additional layers consistently produce well-separated representations regardless of initialization. Low regularization (NB04) achieved the best silhouette of 0.80, while no regularization (NB05) showed slightly worse clustering despite similar accuracy.

**Next steps:** The next steps are to analyze and synthesize the results. 

### **Research question revisited:**

Does spacing out clustering in the neural network layer outputs as an embedding space improve representations and increase classification accuracy?

-   Simply put, NO. Even though we visually saw the embedding space cluster extremly tight together in the final versions, models automatically learn to cluster on their own, achieving higher silhouette scores than with manual clustering intervention. This supports the idea that clustering is important for high accuracy, but demonstrates that forcing it is not the right way to achieve high silhoutte and high accuracy. 

### Error Analysis Findings

I ran both the best centroid model (NB09, test 95.61%) and the best simplex model (NB02, test 96.54%) on the full MNIST test set (10,000 samples) and compared their errors.

**Error counts:** Centroid made 439 errors, Simplex made 346 errors. Of these, 265 samples were misclassified by *both* models, 174 were centroid-only errors, and 81 were simplex-only errors. 9,480 samples were correctly classified by both.

**Common error patterns:**

1. **Digits 9 and 8 are the hardest for both models.** Digit 9 had 53 shared errors (the most of any class) and the lowest recall for both models (centroid 93.0%, simplex 93.4%). Digit 8 had 37 shared errors and the second-lowest recall (centroid 93.3%, simplex 95.4%). These digits have the most ambiguous handwriting variation.

2. **The confused pairs match EDA predictions.** The confusion matrices show the highest off-diagonal counts in the digit pairs identified during EDA — 4/9, 3/5, 7/9, and 2/7. These pairs share similar stroke geometry (e.g., open vs closed loops in 9 vs 4, crossbar presence in 7 vs 2).

3. **When both models are wrong, they usually agree on the wrong answer.** Of the 265 shared errors, 221 (83%) had both models predicting the *same* wrong class. Only 44 cases had different wrong predictions. This suggests these errors come from genuinely ambiguous samples rather than random model noise.

**Why do you think these errors occur?**

The shared errors are concentrated in digits with overlapping stroke geometry — a 9 with an open top looks like a 4, a poorly written 3 looks like a 5 or 8, and a 7 without a clear crossbar looks like a 1. The fact that both architecturally different models make the same mistakes (83% agreement on the wrong class) suggests these are near the decision boundary of what the handwriting actually represents. Some may even be labeling errors in the dataset.

**What could potentially fix these errors?**

A convolutional architecture (CNN) would likely fix many of these errors since it can capture local spatial features (loops, crossbars, stroke endpoints) that a fully-connected MLP flattens away. Data augmentation (small rotations, elastic deformations) could also help by exposing the model to more writing variation during training. For the genuinely ambiguous samples, ensemble methods combining both models could help — since 174 centroid errors and 81 simplex errors are unique to each model, an ensemble could potentially correct those by majority vote.

### Key Insights

**What techniques had the biggest impact on performance?**

The simplex model where fixed centroids were used for training improved the clustering on "too regularized models." 

**What surprised you about this problem?**

The thing that surprised me the most about this problem was that in the end, the neural network without any clustering help achieved the highest accuracy and hihest silhoutte score. Looking back I should have started here, and better understood silhoutte in the neural networks before tring to manipulate the embedding space. 

**If you had more time, what would you try next?**

If I had more time I would run the same experiments on more difficult daitasets to see if the results were the same. 
