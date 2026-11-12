PYTHON ?= python3

.PHONY: check
check:
	$(PYTHON) -B private_evaluator/grade.py validate
	$(PYTHON) -B private_evaluator/grade.py controls
	PYTHONPATH=private_evaluator $(PYTHON) -B -m unittest discover -s tests -v
