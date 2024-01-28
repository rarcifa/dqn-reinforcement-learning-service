run-local:
	poetry run dqn-reinforcement-learning-local

lint:
	flake8 --statistics --show-source --benchmark --config .flake8 dqn-reinforcement-learning