# Evaluation
***
The **Kendall's Tau coefficient** to assess the ranking quality.

Kendall's Tau is a statistic used to measure the ordinal association between two measured quantities. A Tau value close to 1 indicates strong agreement between two rankings, while a value close to -1 suggests strong disagreement. A value around 0 implies no correlation.

### Formula
#### Kendall’s Tau = (C – D) /( C + D)
Where C is the number of concordant pairs and D is the number of discordant pairs.

**Reference**
* https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html
* https://www.statology.org/kendalls-tau/