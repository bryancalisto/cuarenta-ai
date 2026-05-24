run_imitation_ui:
	python3 src/imitation_learning/train.py --checkpoint data/policies/cuarenta_policy_final.pt
train_ppo:
	python3 src/ppo.py --opponent-mode selfplay --resume checkpoints/ckpt_02052096.pt