Surprise score is used to capture the relative change for the root cause item.
$$S_i(m) = 0.5(  p_i\log_2(\frac{2p_i} {p_i+q_i}) + q_i \log_2(\frac{2q_i}{p_i+q_i}) )$$
$$p_i(m)= \frac{F_i(m)} {F(m)} $$
$$q_i(m)= \frac{A_i(m)} {A(m)} $$
where $F_i$ is the forecasted value for root cause item $i$,  $A_i$ is the actual value for root cause item $i$, $F$ is the forecasted value for the anomly point and  $A$ is the actual value for anomaly point.
For details of the surprise score, refer to [this document](https://www.usenix.org/system/files/conference/nsdi14/nsdi14-paper-bhagwan.pdf)
