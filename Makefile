# --debug=basic
# disable built-in rules
.SUFFIXES:



PYTHON = ~/anaconda/bin/python

WORKING = ../data/working

ALL += $(WORKING)/parcels-features-census_tract.csv
ALL += $(WORKING)/parcels-features-zip5.csv
ALL += $(WORKING)/transactions-al-g-sfr.csv
ALL += $(WORKING)/transactions-subset-test.csv   # proxy for -train -validate -train+validate

all: $(ALL)

$(WORKING)/parcels-features-census_tract.csv: parcels-features.py
	$(PYTHON) parcels-features.py --geo census_tract

$(WORKING)/parcels-features-zip5.csv: parcels-features.py
	$(PYTHON) parcels-features.py --geo zip5

$(WORKING)/transactions-al-g-sfr.csv: transactions.py \
	$(WORKING)/parcels-features-census_tract.csv $(WORKING)/parcels-features-zip5.csv
	$(PYTHON) transactions.py

$(WORKING)/transactions-subset-test.csv: transactions-subset.py
	$(PYTHON) transactions-subset.py
