# Round 1

## Imitation learning
First, I trained a policy with imitation learning using DAGGER (Dataset Aggregation). I chose DAGGER as it's an effective and easy to undestand algorithm. Good enough for the job of giving initial "knowledge" to the cuarenta agent in order to not start the reinforcement learning loop from random weights (which would increase the training time).

I played 10 games of cuarenta with up to 40 steps. After each 40 steps, the DAGGER algorithm would run a training batch.

## Reinforcement learning with PPO
Later, I trained the policy with PPO (Proximal Policy Optimization) by self-play starting on the final imitation learning policy (data/policies/cuarenta_policy_final.pt).

**SPECS:**
- 3,000,000 steps.
- 1e-3 (up to iteration / snapshot 53) and 1e-4 (from there on) learning rate.
- 8 concurrent environments.
- 4 epochs.
- batch size of 128.
- gamma of 0.99.
- For opponent selection, use a 20% probability of selecting the latest snapshot. A new opponent is selected on snapshot save.

This phase took ~6 hours, ~1 hour per 500,000 steps on average.

## Results
I didn't have metric scores collection code set up when I ran round 1 up to snapshot 41.

Starting from snapshot 42 (iteration 1,050,000) got these metrics:

| snapshot | value loss           | entropy                      | win rate            | episode reward    |   |   |   |   |   |
|----------|----------------------|------------------------------|---------------------|-------------------|---|---|---|---|---|
| 42       | 2 - 3, very unstable | 0.16 - 0.125,  very unstable | 0.7 - 0.4, unstable | 20 - 30, unstable |   |   |   |   |   |
| 53       | 3 - 4.25             | 0.12 - 0.95                  | 0.6 - 0.4           | 32 - 39           |   |   |   |   |   |
| 64       | 4 - 4                | 0.10 - 0.07                  | 0.14 - 0.27         | 40 - 39.5         |   |   |   |   |   |
| 75       | 5 - 3 - 5            | 0.065 - 0.057                | 0.07 - 0.35         | 67.5              |   |   |   |   |   |
|          |                      |                              |                     |                   |   |   |   |   |   |
|          |                      |                              |                     |                   |   |   |   |   |   |
|          |                      |                              |                     |                   |   |   |   |   |   |
|          |                      |                              |                     |                   |   |   |   |   |   |
|          |                      |                              |                     |                   |   |   |   |   |   |



To test the model in the real world, two humans played four games against two agents running snapshot 75. Humans won the four games. The agents didn't make any obvious mistakes. They didn't make any evident planned plays, though.

The games were taking long. Up to ~180 steps. There was no much "cartón" captured. It seems like the agents are fit for long games.
