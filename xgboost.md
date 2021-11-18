# XGBOOST

{% embed url="https://medium.com/sfu-cspmp/xgboost-a-deep-dive-into-boosting-f06c9c41349" %}

## parallelization

\
not parallelize to generate TREES, instead, it generates different BRANCHES in parallel using openMP, a shared memory multi-threading algo\
the general architecture is inherited from Boosting - sequentially generate trees; \
xgboost corrects the error by residuals. \
[https://stackoverflow.com/a/34166401/15725337](https://stackoverflow.com/a/34166401/15725337)
