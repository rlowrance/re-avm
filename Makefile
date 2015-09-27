# --debug=basic
# disable built-in rules
.SUFFIXES:



PYTHON = ~/anaconda/bin/python

WORKING = ../data/working

ALL += $(WORKING)/transactions-al-g-sfr.csv
ALL += $(WORKING)/transactions-subset-test.csv   # proxy for -train -validate -train+validate

all: $(ALL)

$(WORKING)/transactions-al-g-sfr.csv: transactions.py
	$(PYTHON) transactions.py --input ~/Dropbox/real-estate-los-angeles/

$(WORKING)/transactions-subset-test.csv: transactions-subset.py
	$(PYTHON) transactions-subset.py
