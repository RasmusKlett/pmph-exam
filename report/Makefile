### VARS
OUT_DIR=texout
IN_DIR=input/content

# Name - the latex main document
TEX_MAIN=main

# PDF-file name
COURSE=pmph
ASSIGNMENT=exam
VERSION=1
PDF_TITLE=$(COURSE)-$(ASSIGNMENT)-v$(VERSION)

all: main.pdf

# MAIN LATEXMK RULE

main.pdf:	$(TEX_MAIN).tex $(IN_DIR)/*
			latexmk -pdf -jobname=$(PDF_TITLE) -outdir=$(OUT_DIR) $(TEX_MAIN).tex

spellcheck: $(IN_DIR)/*
			find $(IN_DIR) -maxdepth 1 -type f -name "*.tex" -exec \
				aspell \
					--conf="./aspell/aspell.conf" \
					--extra-dicts="./aspell/soma.pws" \
					check {} \;

cleanin:
			find $(IN_DIR) -maxdepth 1 -type f -name "*.bak" -delete

cleanout:
			rm $(OUT_DIR)/*

cleanall:	cleanin cleanout

.PHONY: main.pdf all clean spellcheck cleanin cleanout cleanall
