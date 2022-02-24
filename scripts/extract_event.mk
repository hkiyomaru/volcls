SCRIPT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
ROOT_DIR := $(SCRIPT_DIR)/..
SEED_LEXICON_DIR := $(ROOT_DIR)/data/seed_lexicon

LANG :=
DATA_DIR :=
DATA_EXT := txt.gz
RESULT_DIR :=
NUM_FILES := 100000
PYTHON := $(shell which python)

VOL_SEED_DICT_FILE := $(SEED_LEXICON_DIR)/volitionality.$(LANG).txt
ANI_SEED_DICT_FILE := $(SEED_LEXICON_DIR)/animacy.$(LANG).txt

EXTRACTOR := $(ROOT_DIR)/src/extract_event.py

NICE := 19
DATAS := $(shell find $(DATA_DIR) -name "*.$(DATA_EXT)" | head -n $(NUM_FILES))
RESULTS := $(patsubst $(DATA_DIR)%.$(DATA_EXT),$(RESULT_DIR)%.touch,$(DATAS))

all: $(RESULTS)

$(RESULTS): $(RESULT_DIR)%.touch: $(DATA_DIR)%.$(DATA_EXT)
	mkdir -p $(dir $@) && \
	nice -n $(NICE) $(PYTHON) $(EXTRACTOR) --data_file $< --output_dir $(dir $@) --vol_seed_lex_file $(VOL_SEED_DICT_FILE) --ani_seed_lex_file $(ANI_SEED_DICT_FILE) --lang $(LANG) && \
	touch $@
