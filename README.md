# Coding-Test-for-PhD
# Reproduction of Two Papers on Constrained Sampling and Adversarial Robustness

This project contains the reproduction of two papers:

### 1. Constrained Sampling with Primal-Dual Langevin Monte Carlo
**Citation**:
<pre><code class="language-bibtex">@InProceedings{Chamon24c, 
  author = "Chamon, L. F. O. and Jaghargh, M. R. K. and Korba, A.", 
  title = "Constrained sampling with primal-dual {L}angevin {M}onte {C}arlo", 
  booktitle = "Conference on Neural Information Processing Systems (NeurIPS)", 
  year = "2024", 
  } </code></pre>

### 2. Adversarial Robustness with Semi-Infinite Constrained Learning
**Citation**:
<pre><code class="language-bibtex">@article{robey2021adversarial,
title={Adversarial robustness with semi-infinite constrained learning},
author={Robey, Alexander and Chamon, Luiz and Pappas, George J and Hassani, Hamed and Ribeiro, Alejandro},
journal={Advances in Neural Information Processing Systems},
volume={34},
pages={6198--6215},
year={2021}
}</code></pre>


---

## Hardware Note

- The first paper was run on CPU.
- The second paper was run on an A6000 GPU. Due to limited GPU resources in my group and insufficient capabilities of free Colab, I borrowed an A6000 GPU from a friend. However, I only had short-term access, so results may differ significantly from the original paper due to the unsufficient hyperparameter tuning. I apologize for this.
Despite the differences, I tried to analyze the results and find out what triggered the results. Also, since there were other people using the same device, some results might not be successfully saved, thus I might use some screenshot and some approximation results.

Additionally, I implemented all code using Functions rather than Classes for personal convenience. While the original source code of the second paper is written using Classes, I did not need to reproduce other baseline methods and thus decided not to use Classes (if I need to compare then I'll consider a code structure similar to the source code, i.e., using Classes).
Moreover, I wrote .ipynb files rather than .py files for a better visualization.

---

## Result Report

### Paper 1: Constrained Sampling with Primal-Dual Langevin Monte Carlo

I reproduced the PD-LMC, Projected LMC, and Rejection Sampling algorithms in PyTorch. My implementation was guided by the original source code in terms of:
- Function interfaces of all three algorithms
- Implementation details such as hyperparameters
- Overall structure of the algorithms

The results are reported below:
####  Runtime:
- Rejection Sampling: **1.57 seconds**
- Projected LMC: **45.27 seconds**
- PD-LMC: **122.05 seconds**

####  Hyperparameters selection:
The best estimation is obtained based on:
- Step size (x): `5e-4`
- Step size (λ): `0.1`

####  Mean Estimation and Constraint Violation
| Algorithm       | Mean                 | Out-of-Ellipsoid Rate |
|----------------|----------------------|------------------------|
| Rejection (true)| 0.0001, 1.4156       | N/A                    |
| Proj. LMC       | 0.0002, 0.7415       | On Ellipsoid: 7.59%    |
| PD-LMC          | 0.0030, 1.4137       | Outside: 4.16%, On: 0.09% |

####  Reproduction of Figure 2
As shown below, PD-LMC achieves a more accurate mean estimation and the samples are more similar to the Rejection than Projected LMC.
![PD-LMC Samples](#./samples.png)

---

### Paper 2: Adversarial Robustness with Semi-Infinite Constrained Learning

I reproduced the results from **Table 1** and **Figure 2** of the paper. The implementation referenced the original source code for:
- Use of loss functions: `l_pert`, `l_nom`, `l_ro`
- CIFAR-10 data transforms

I wrote the code for PGD and FGSM attacks based on the paper's descriptions before receiving your email so I did not use any existing code for this.

I experimented with five settings with the following details (other hyperparameters are as in the original paper):

1. **KL divergence** for `l_pert` and `l_ro`, **Cross Entropy** for `l_nom`, typical normalization for CIFAR-10.
![Training Curve](#trainingcurve1.png)
Clean acc: appro. '0.75'
FGSM acc: appro. '0.53'
PGD20 acc: appro. '0.27'

2. Cost function from source for `l_pert`, CE for `l_ro` and `l_nom`,  used full source transforms.
![Training Curve](#trainingcurve2.png)
Clean acc: appro. '0.87'
FGSM acc: appro. '0.53'
PGD20 acc: appro. '0.005'

3. KL divergence for `l_pert` and `l_ro`, CE for `l_nom`, step size for `v` set to `5e-3` due to slow learning (however, this did not work), used full source transforms..
![Training Curve](#trainingcurve2.png)
Clean acc: '0.8825'
FGSM acc: '0.5571'
PGD20 acc: '0.0057'

4. Same as 3, but step size for `v` tuned to `1e-3`, used full source transforms..
![Training Curve](#trainingcurve5.png)
Clean acc: '0.8810'
FGSM acc: '0.5497'
PGD20 acc: '0.0055'

Analysis:
Compare 4 with 1, the poor learning curve of PGD20 accuracy in Setting 4—despite relatively better curves for Clean and FGSM accuracy—appears to be influenced by the choice of data transforms.
Compare 4 with 3, the behavior of the v curve in Setting 4, which differs from the original figure in the paper, does not seem to be explained by the step size of v.
Compare 4 with 2, the choice of KL divergence seems to be better than CE for 'l_ro'.

Improvements and further tuning:
1, KL divergence for `l_pert` and `l_ro`, CE for `l_nom`, Source transforms, plus additional normalization (as in Result 1).
2, The reason behind the cureve of 'v' is still unclear, Possible directions for further investigation:
-Additional tuning of the step size for updating v,
-Check whether there is any misunderstanding for me in the implementation of the update of 'v',
-It's may because of the hardware differences that have influenced learning behavior.

x. Cost function from source for `l_pert`, CE for `l_ro` and `l_nom`, updated `v` per batch iteration (SEEMED to be same as source), used full source transforms.
![Training Curve](#trainingcurvex.png)
Clean acc:  '0.10'
FGSM acc:  '0.10'
PGD20 acc:  '0.10'
Well... at least all the accuracies are equally terrible. Consistency is key, right? :P
---

## Repository Structure
- `paper1.ipynb`: Code for Paper 1.
- `paper2.ipynb`: Code for Paper 2.
- `xxx.png`: Reproduction of figures.


---

