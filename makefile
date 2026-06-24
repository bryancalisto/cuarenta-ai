train_with_imitation:
	python3 src/imitation_learning/train.py --checkpoint data/policies/cuarenta_policy_final.pt

play_with_ai:
	python3 src/imitation_learning/play_vs_agent.py --show-agent-hand true --checkpoint /Users/t1000/Documents/src/40-ai/data/policies/policy_2001920.pt --human-players 0 --delay 2

train_ppo:
	python3 src/ppo.py --steps 2_000_000 --n-envs 10 --lr 1e-3 --opponent-mode selfplay --init-policy data/policies/cuarenta_dagger_policy_final.pt --device cpu
