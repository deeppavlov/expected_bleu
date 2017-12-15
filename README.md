# Differentiable lower bound for expected BLEU score

This is our implementation of lower bound for expected BLEU score. For details, see the corresponding [NIPS'17 workshop paper](https://arxiv.org/abs/1712.04708)

We are using <img src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png" width="100">

Our code inspired on this [tensorflow.NMT BLEU script](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py)
* ```BLEU_results.ipynb``` - results with graphs(TODO: refactor, optimize)

* ``` multiBLEU_results.ipynb ``` - results of batched version of our LB on the same but batched version of toy task.

* ``` bleu.ipynb ``` - the same as ```BLEU_results.ipynb``` but contains only simple example for unigrams (word modified precision)

* ```modules```
* * ```expectedMultiBleu.py``` - contains implementation of batched lower bound version from the paper. We recommend to read this (instead of expectedBLEU.py) since this is documentated.
* * ``` modules/expectedBLEU.py ``` - contains implementation of lower bound from paper  

* * ``` modules/matrixBLEU.py ``` - implements computation of BLEU in matrix form (for degenerate distribution it coincides with  BLEU)
* * ``` modules/toyExperiment.py ``` - here you can find implementation of the toy example  as described in paper
* * ``` modules/utils.py ``` - support functions
### BLEU3
<img src="https://raw.githubusercontent.com/deepmipt/expected_bleu/master/images/BLEU3.png" width="720">

NOTE: to reduce complexity of our algorithm we iterate only over words from reference (since all other terms in sum are zeros)
