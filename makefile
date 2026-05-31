run_imitation_ui:
	python3 src/imitation_learning/train.py --checkpoint data/policies/cuarenta_policy_final.pt

play_with_ai:
	python3 src/imitation_learning/play_vs_agent.py --checkpoint data/policies/cuarenta_policy_iter24.pt --human-players 0,2 --delay 2.5

train_ppo:
	python3 src/ppo.py --steps 1_000_000 --lr 1e-3 --opponent-mode selfplay --init-policy data/policies/cuarenta_policy_iter11.pt --device mps
