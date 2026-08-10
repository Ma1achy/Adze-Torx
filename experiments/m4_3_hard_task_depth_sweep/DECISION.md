# M4.3 decision

## M4_3_Q_NEGATIVE

The 120-step comparison was correctly rejected as optimization-limited because
all Q values were still improving materially. A common 240-step budget was then
run for every Q, task depth, core type, and seed.

At k=8 and k=12, nonlinear Q=8 and Q=12 were at least 5% worse than nonlinear
Q=1, with paired 95% confidence intervals excluding zero. The same degradation
appeared in the affine control. Deep-Q trajectories improved intermediate
states, but no deep-Q configuration achieved a compensating 5% best-intermediate
improvement or endpoint improvement.

This is a robust Q-negative result for the tested M4.2 public-Torx affine-plus-
nonlinear residual core and harder operator-composition task. It does not make a
universal claim about other recurrent architectures. The depth investigation
stops here; do not begin M5 automatically.
