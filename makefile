run_imitation_ui:
	python3 src/imitation_learning/train.py --checkpoint data/policies/cuarenta_policy_final.pt
train_ppo:
	python3 src/ppo.py --steps 1_000_000 --lr 2e-4 --opponent-mode selfplay --init-policy data/policies/cuarenta_policy_iter19.pt --device mps