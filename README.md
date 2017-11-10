Results are in ```BLEU_results.ipynb```

Implementation of lower bound from paper you can find in ``` eBLEU_from_paper.py ```


``` eBLEU.py ``` - implement computation bleu in matrix form (for degenerate distribution it coincides with  BLEU)


``` bleu.ipynb ``` - the same as ```BLEU_results.ipynb``` but contains only example for unigrams (word modified precision)


NOTE: to reduce complexity of out algorithm we iterate only over words from reference (since all other terms in sum are zeros)
