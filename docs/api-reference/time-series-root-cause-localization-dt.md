At Mircosoft, we develop a decision tree based root cause localization method which helps to find out the root causes for an anomaly incident incrementally. 

## Multi-Dimensional Root Cause Localization
It's a common case that one measure are collected with many dimensions (*e.g.*, Province, ISP) whose values are categorical(*e.g.*, Beijing or Shanghai for dimension Province). When a measure's value deviates from its expected value, this measure encounters anomalies. In such case, operators would like to localize the root cause dimension combinations rapidly and accurately. Multi-dimensional root cause localization is critical to troubleshoot and mitigate such case.

## Algorithm

The decision based root cause localization method is unsupervised, which means training step is no needed. It consists of the following major steps:
(1) Find best dimension which divides the anomaly and unanomaly data based on decision tree according to entropy gain and entropy gain ratio.
(2) Find the top anomaly points for the selected best dimension.

### Decision Tree

[Decision tree](https://en.wikipedia.org/wiki/Decision_tree) algorithm chooses the highest information gain to split or construct a decision tree.  We use it to choose the dimension which contributes the most to the anomaly. Following are some concepts used in decision tree.

#### Information Entropy

Information [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) is a measure of disorder or uncertainty. You can think of it as a measure of purity as well.The less the value , the more pure of data D. 

$$Ent(D) = - \sum_{k=1}^{|y|}  p_k\log_2(p_k) $$

where $p_k$ represents the probability of an element in dataset. In our case, there are only two classed, the anomaly points and the normaly points.  $|y|$ is the count of total anomalies.

#### Information Gain
[Information gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees) is a metric to measure the reduction of this disorder in our target class given additional information about it. Mathematically it can be written as:

$$Gain(D, a) = Ent(D) - \sum_{v=1}^{|V|} \frac{|D^V|}{|D |} Ent(D^v) $$

Where $Ent(D^v)$ is the entropy of set points in D for which dimension $a$ is equal to $v$, $|D|$ is the total number of points in dataset $D$.  $|D^V|$ is the total number of points in dataset $D$ for which dimension $a$ is equal to $v$.

For all aggregated dimensions, we calculate the information for each dimension. The greater the reduction in this uncertainty, the more information is gained about D from dimension $a$.

#### Entropy Gain Ratio

Information gain is biased toward variables with large number of distinct values. A modification is [information gain ratio](https://en.wikipedia.org/wiki/Information_gain_ratio), which reduces its bias.

$$Ratio(D, a) = \frac{Gain(D,a)} {IV(a)} $$

where intrinsic value(IV) is the entropy of split (with respect to dimension $a$ on focus).

$$IV(a) = -\sum_{v=1}^V\frac{|D^v|} {|D|} \log_2 \frac{|D^v|} {|D|}  $$

In out strategy, firstly, for all the aggration dimensions, we loop all the dimensions to find the dimension who's entropy gain is above mean entropy gain ration, then from the filtered dimensions,  we select the dimension with highest entropy ratio as the best dimension. In the meanwhile, dimensions for which the anomaly value count is only one, we include it when calculation.

> [!Note]
> 1. As our algorithm depends on the data you input, so if the input points is incorrect or incomplete, the calculated result will be unexpected. 
> 2. Currently, the algorithm localize the root cause incrementally, which means at most one dimension with the values are detected. If you want to find out all the dimension that contributes to the anomaly, you can call this API recursively. 
