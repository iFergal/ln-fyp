# Lightning Network Simulator
Intended to show the benefits of atomic multi-path payments on the Lightning Network and its impact on the success rate of payments.

## Roadmap
#### Essential
- Create basic network/graph of nodes, with edges representing payment channels of equity and allow for (attempt at least) transfer of regular payments between nodes.
- Impose fees for hops (base + percentage) on nodes based on some intuitive fee distribution.
- Fix routing: should be based on fees + time-locks (if time-locks implemented), and maybe a margin on equity if available, not lowest equity path!
  - Should also find path in manner different to reducing to sub-graph based on equity.
- Define topology creating functions for small-world & scale-free.
- Apply a relevant distribution of initial channel balances & equities.
- Apply a distribution of rate of payments across nodes.
- Implement AMP payments.
  - This will require some insight & trial+error into how to effectively split payments - which leads to further evaluation.
- Create a config file that updates network configuration variables.
- Record relevant events for statistical analysis.
- Fine tune AMP payment splitting process based on statistical analysis.
- Implement various proposed improvements on AMP such as messaging ahead to gain insight into lowest equity in given path direction, before attempting payment.

#### Non-essential
- Use different routing methods to see if any anomalies occur.
