# --debug=basic
# disable built-in rules
.SUFFIXES:



PYTHON = ~/anaconda/bin/python

WORKING = ../data/working

ALL += $(WORKING)/census-features-derived.csv
ALL += $(WORKING)/chart-01.txt

ALL += $(WORKING)/parcels-features-census_tract.csv
ALL += $(WORKING)/parcels-features-zip5.csv

#ALL += $(WORKING)/samples-test.csv   # proxy for -train -train-test -train-train
#ALL += $(WORKING)/samples-train.csv   # proxy for -train -train-test -train-train
#ALL += $(WORKING)/samples-train-validate.csv   # proxy for -train -train-test -train-train
#ALL += $(WORKING)/samples-validate.csv   # proxy for -train -train-test -train-train

ALL += $(WORKING)/summarize-df-samples-train.csv
ALL += $(WORKING)/transactions-al-g-sfr.csv


.PHONY : all
all: $(ALL)

.PHONY : parcels-features
parcels-features: $(WORKING)/parcels-features-census_tract.csv $(WORKING)/parcels-features-zip5.csv

$(WORKING)/census-features-derived.csv: census-features.py layout_census.py
	$(PYTHON) census-features.py

$(WORKING)/chart-01.txt: chart-01.py $(WORKING)/chart-01.data.pickle
	$(PYTHON) chart-01.py
	
$(WORKING)/chart-01.data.pickle: chart-01.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) chart-01.py --data

$(WORKING)/parcels-features-census_tract.csv: parcels-features.py layout_parcels.py
	$(PYTHON) parcels-features.py --geo census_tract

$(WORKING)/parcels-features-zip5.csv: parcels-features.py layout_parcels.py
	$(PYTHON) parcels-features.py --geo zip5

$(WORKING)/transactions-al-g-sfr.csv: transactions.py \
	$(WORKING)/census-features-derived.csv \
	$(WORKING)/parcels-features-census_tract.csv $(WORKING)/parcels-features-zip5.csv 
	$(PYTHON) transactions.py

$(WORKING)/samples-test%csv $(WORKING)/samples-train%csv $(WORKING)/samples-train-validate%csv $(WORKING)/samples-validate%csv: samples.py $(WORKING)/transactions-al-g-sfr.csv
	$(PYTHON) samples.py

$(WORKING)/summarize-samples-train.csv: summarize-df.py summarize.py
	$(PYTHON) summarize-df.py --in samples-train.csv 
