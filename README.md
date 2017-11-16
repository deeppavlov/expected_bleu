# Differentiable lower bound for expected BLEU score

* ```BLEU_results.ipynb``` - results with graphs

* ``` bleu.ipynb ``` - the same as ```BLEU_results.ipynb``` but contains only simple example for unigrams (word modified precision)

* ```modules```
* * ``` modules/expectedBLEU.py ``` - contains implementation of lower bound from paper  

* * ``` modules/matrixBLEU.py ``` - implements computation of BLEU in matrix form (for degenerate distribution it coincides with  BLEU)
* * ``` modules/toyExperiment.py ``` - here you can find implementation of the toy example  as described in paper
* * ``` modules/utils.py ``` - support functions
### BLEU3
![BLEU3](https://raw.githubusercontent.com/deepmipt/expected_bleu/master/images/BLEU3.png)

NOTE: to reduce complexity of out algorithm we iterate only over words from reference (since all other terms in sum are zeros)
