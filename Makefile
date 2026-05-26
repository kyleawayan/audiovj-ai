# audiovj-ai pipeline

.PHONY: analyze import preprocess train evaluate evaluate-pipeline e2e clean-stems help

MUSIC_DIR  ?= $(HOME)/audiovj-data/music_to_analyze
STRUCT_DIR ?= data/struct
EPOCHS     ?= 50

help:
	@echo "Targets:"
	@echo "  analyze            allin1 analyze every audio file in MUSIC_DIR"
	@echo "  import             parse JSONs into Track records"
	@echo "  preprocess         extract mel features per track"
	@echo "  train              train phrase predictor (EPOCHS=$(EPOCHS))"
	@echo "  evaluate           raw model metrics"
	@echo "  evaluate-pipeline  raw + state-manager aggregate"
	@echo "  e2e                run the whole chain in order"
	@echo "  clean-stems        rm demix/ and /tmp/audiovj-allin1/"
	@echo ""
	@echo "Vars (override via 'make <t> VAR=value'):"
	@echo "  MUSIC_DIR=$(MUSIC_DIR)"
	@echo "  STRUCT_DIR=$(STRUCT_DIR)"
	@echo "  EPOCHS=$(EPOCHS)"

analyze:
	uv run audiovj analyze-folder $(MUSIC_DIR) --struct-dir $(STRUCT_DIR)

import:
	uv run audiovj import-folder $(MUSIC_DIR) --struct-dir $(STRUCT_DIR)

preprocess:
	uv run audiovj preprocess

train:
	uv run audiovj train --epochs $(EPOCHS)

evaluate:
	uv run audiovj evaluate

evaluate-pipeline:
	uv run audiovj evaluate-pipeline

e2e: analyze import preprocess train evaluate evaluate-pipeline

clean-stems:
	rm -rf demix/ /tmp/audiovj-allin1/
