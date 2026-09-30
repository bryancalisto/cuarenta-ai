train_with_imitation:
	python3 src/imitation_learning/train.py

play_with_ai:
	python3 src/imitation_learning/play_vs_agent.py --show-agent-hand true --checkpoint /Users/t1000/Documents/src/40-ai/output/policies/lstm_6/policy_1003520.pt --human-players 0 --delay 2

train_ppo_without_lstm:
	python3 src/ppo.py --use-lstm false --steps 2_000_000 --n-envs 10 --lr 1e-3 --opponent-mode selfplay --init-policy output/policies/cuarenta_dagger_policy_final.pt --device cpu --save-dir output/policies/ppo0/no_lstm

train_ppo_with_lstm_1:
	python3 src/ppo.py --use-lstm true --lstm-hidden 128 --steps 2_000_000 --n-envs 10 --lr 1e-3 --opponent-mode selfplay --init-policy output/policies/cuarenta_dagger_policy_final.pt --device cpu --save-dir output/policies/ppo0/lstm_1

train_ppo_with_lstm_2:
	python3 src/ppo.py --use-lstm true --lstm-hidden 256 --steps 2_000_000 --n-envs 10 --lr 1e-3 --opponent-mode selfplay --init-policy output/policies/cuarenta_dagger_policy_final.pt --device cpu --save-dir output/policies/ppo0/lstm_2

train_ppo_with_lstm_3:
	python3 src/ppo.py --use-lstm true --lstm-hidden 128 --steps 2_000_000 --n-envs 10 --lr 1e-3 --opponent-mode selfplay --init-policy output/policies/cuarenta_dagger_policy_final.pt --device cpu --save-dir output/policies/ppo0/lstm_3

train_ppo_with_lstm_4:
	python3 src/ppo.py --use-lstm true --lstm-hidden 128 --trunk-hidden 512 --value-hidden 256 --steps 2_000_000 --n-envs 10 --lr 3e-3 --opponent-mode selfplay --init-policy none --device cpu --save-dir output/policies/ppo0/lstm_4

train_ppo_with_lstm_5:
	python3 src/ppo.py --use-lstm true --lstm-hidden 64 --trunk-hidden 64 --value-hidden 64 --steps 2_000_000 --n-envs 10 --lr 1e-3 --opponent-mode selfplay --init-policy none --device cpu --save-dir output/policies/ppo0/lstm_5

train_ppo_with_lstm_6:
	python3 src/ppo.py --use-lstm true --lstm-hidden 128 --steps 2_000_000 --n-envs 10 --lr 1e-3 --opponent-mode selfplay --init-policy output/policies/cuarenta_dagger_policy_final.pt --device cpu --save-dir output/policies/lstm_6

train_ppo_with_lstm_7:
	python3 src/ppo.py --use-lstm true --lstm-hidden 128 --steps 2_000_000 --n-envs 10 --lr 1e-3 --opponent-mode selfplay --init-policy none --device cpu --save-dir output/policies/lstm_7