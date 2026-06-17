# Image captioning — run from the repo root (artifacts still live under app/).
# Override defaults: make train DATASETS=flickr30k EPOCHS=30 YES=1

ROOT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
APP_DIR := $(ROOT_DIR)app
VENV_PYTHON := $(ROOT_DIR).venv/bin/python
PYTHON3_12 ?= python3.12
PYTHON := $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python3)

DATASETS ?= flickr8k
EPOCHS ?= 10
PORT ?= 8501
ADDRESS ?= localhost

.PHONY: help install check list-datasets train train-retrain evaluate extract-features \
	venv gpu-metal train-demo evaluate-demo \
	streamlit streamlit-headless streamlit-app clean clean-dry-run

help:
	@echo "Image captioning — Makefile targets (repo root)"
	@echo ""
	@echo "Setup:"
	@echo "  make venv                 create .venv with python3.12 (if missing)"
	@echo "  make gpu-metal            install tensorflow==2.19.1 + tensorflow-metal==1.2.0"
	@echo "  make install              pip install -r requirements.txt"
	@echo "  make check                verify Python and TensorFlow import"
	@echo ""
	@echo "Pipeline (wraps: cd app && python main.py …):"
	@echo "  make list-datasets        show dataset ids and download info"
	@echo "  make train                train (DATASETS=$(DATASETS), EPOCHS=$(EPOCHS))"
	@echo "  make train-retrain        train with --force-retrain -y"
	@echo "  make train-demo           demo train (flickr8k,flickr30k, epochs=30, YES=1)"
	@echo "  make evaluate             BLEU on test split (YES=1 to skip prompts)"
	@echo "  make evaluate-demo        demo evaluate (flickr8k,flickr30k, YES=1)"
	@echo "  make extract-features     build/refresh features_*.dump only"
	@echo ""
	@echo "Streamlit UI:"
	@echo "  make streamlit            python main.py streamlit (PORT=$(PORT))"
	@echo "  make streamlit-headless   same, no browser tab (PORT=$(PORT))"
	@echo "  make streamlit-app        streamlit run app.py (PORT=$(PORT), ADDRESS=$(ADDRESS))"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                interactive artifact cleanup"
	@echo "  make clean-dry-run        list deletable items only"
	@echo ""
	@echo "Variables: DATASETS EPOCHS PORT ADDRESS YES=1"

install:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -r $(ROOT_DIR)requirements.txt

venv:
	@if [ -d "$(ROOT_DIR).venv" ]; then \
		echo ".venv already exists (using $(VENV_PYTHON) if available)."; \
	else \
		echo "Creating .venv with $(PYTHON3_12)"; \
		$(PYTHON3_12) -m venv $(ROOT_DIR).venv; \
	fi

gpu-metal:
	$(PYTHON) -m pip install "tensorflow==2.19.1" "tensorflow-metal==1.2.0"
	$(PYTHON) -c "import tensorflow as tf; print('tensorflow', tf.__version__)"

check:
	@$(PYTHON) --version
	@$(PYTHON) -c "import tensorflow as tf; print('tensorflow', tf.__version__)"

list-datasets:
	cd $(APP_DIR) && $(PYTHON) main.py list-datasets

train:
	cd $(APP_DIR) && $(PYTHON) main.py train --datasets $(DATASETS) --epochs $(EPOCHS) \
		$(if $(YES),-y,)

train-retrain:
	cd $(APP_DIR) && $(PYTHON) main.py train --datasets $(DATASETS) --epochs $(EPOCHS) \
		--force-retrain -y

train-demo:
	$(MAKE) train DATASETS=flickr8k,flickr30k EPOCHS=30 YES=1

evaluate:
	cd $(APP_DIR) && $(PYTHON) main.py evaluate --datasets $(DATASETS) \
		$(if $(YES),-y,)

evaluate-demo:
	$(MAKE) evaluate DATASETS=flickr8k,flickr30k YES=1

extract-features:
	cd $(APP_DIR) && $(PYTHON) main.py extract-features --datasets $(DATASETS) \
		$(if $(YES),-y,)

streamlit:
	cd $(APP_DIR) && $(PYTHON) main.py streamlit --port $(PORT)

streamlit-headless:
	cd $(APP_DIR) && $(PYTHON) main.py streamlit --port $(PORT) --headless

streamlit-app:
	cd $(APP_DIR) && $(PYTHON) -m streamlit run app.py \
		--server.port $(PORT) --server.address $(ADDRESS)

clean:
	cd $(APP_DIR) && $(PYTHON) main.py clean $(if $(YES),-y,)

clean-dry-run:
	cd $(APP_DIR) && $(PYTHON) main.py clean --dry-run
