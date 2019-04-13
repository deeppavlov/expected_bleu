# WARNING
Latest version is here [expected_bleu](https://github.com/deepmipt/diff_beam_search/tree/master/expected_bleu) (some bugs were fixed, pytorch 0.4 version)

# Differentiable lower bound for expected BLEU score

This is our implementation of lower bound for expected BLEU score. For details, see the corresponding [NIPS'17 workshop paper](https://arxiv.org/abs/1712.04708)

We are using <img src="https://raw.githubusercontent.com/pytorch/pytorch/master/docs/source/_static/img/pytorch-logo-dark.png" width="100">

Our code inspired on this [tensorflow.NMT BLEU script](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py)

* ``` multiBLEU_results.ipynb ``` - results of batched version of our LB on the same but batched version of toy task.
* ```modules```
* * ```modules/expectedMultiBleu.py``` - contains implementation of batched lower bound version from the paper.
* * ``` modules/expectedBLEU.py ``` - contains implementation of lower bound from paper(no batches)

* * ``` modules/matrixBLEU.py ``` - implements computation of BLEU in matrix form (for degenerate distribution it coincides with  BLEU)
* * ``` modules/utils.py ``` - support functions
### BLEU4
<img src="https://raw.githubusercontent.com/deepmipt/expected_bleu/master/images/BLEU4.png" width="720">
