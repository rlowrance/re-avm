# --debug=basic
# disable built-in rules
.SUFFIXES:



PYTHON = ~/anaconda/bin/python

WORKING = ../data/working

ALL += $(WORKING)/census-features-derived.csv
ALL += $(WORKING)/chart-01.txt

# CHART02 and RFBOUND are obsoleted by RFVAL
# their rules and recipes are in rfbound.mk

RFVAL += $(WORKING)/rfval/200402.pickle
RFVAL += $(WORKING)/rfval/200405.pickle
RFVAL += $(WORKING)/rfval/200408.pickle
RFVAL += $(WORKING)/rfval/200411.pickle
RFVAL += $(WORKING)/rfval/200502.pickle
RFVAL += $(WORKING)/rfval/200505.pickle
RFVAL += $(WORKING)/rfval/200508.pickle
RFVAL += $(WORKING)/rfval/200511.pickle
RFVAL += $(WORKING)/rfval/200602.pickle
RFVAL += $(WORKING)/rfval/200605.pickle
RFVAL += $(WORKING)/rfval/200608.pickle
RFVAL += $(WORKING)/rfval/200611.pickle
RFVAL += $(WORKING)/rfval/200702.pickle
RFVAL += $(WORKING)/rfval/200705.pickle
RFVAL += $(WORKING)/rfval/200708.pickle
RFVAL += $(WORKING)/rfval/200711.pickle
RFVAL += $(WORKING)/rfval/200802.pickle
RFVAL += $(WORKING)/rfval/200805.pickle
RFVAL += $(WORKING)/rfval/200808.pickle
RFVAL += $(WORKING)/rfval/200811.pickle
RFVAL += $(WORKING)/rfval/200902.pickle
ALL += $(RFVAL)

# use max_depth as a proxy for both max_depth and max_features
# use 2004-02 as a proxy for all years YYYY and months MM
CHART03 += $(WORKING)/chart-03/max_depth-2004-02.pdf
ALL += $(CHART03)

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

# chart-01
$(WORKING)/chart-01.txt: chart-01.py $(WORKING)/chart-01.data.pickle
	$(PYTHON) chart-01.py
	
$(WORKING)/chart-01.data.pickle: chart-01.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) chart-01.py --data

# chart-03
#    max_depth is a proxy for both max_depth and max_features
#    2004-02 is a proxy for all years YYYY and all months MM
CHART03REDUCTION = $(WORKING)/chart-03/data.pickle

$(CHART03REDUCTION): chart-03.py $(RFVAL)
	$(PYTHON) chart-03.py --data

$(WORKING)/chart-03/max_depth-2004-02.pdf: chart-03.py $(CHART03REDUCTION)
	$(PYTHON) chart-03.py 

# rfval
$(WORKING)/rfval/200402.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200402

$(WORKING)/rfval/200405.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200405

$(WORKING)/rfval/200408.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200408

$(WORKING)/rfval/200411.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200411

$(WORKING)/rfval/200502.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200502

$(WORKING)/rfval/200505.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200505

$(WORKING)/rfval/200508.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200508

$(WORKING)/rfval/200511.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200511

$(WORKING)/rfval/200602.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200602

$(WORKING)/rfval/200605.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200605

$(WORKING)/rfval/200608.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200608

$(WORKING)/rfval/200611.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200611

$(WORKING)/rfval/200702.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200702

$(WORKING)/rfval/200705.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200705

$(WORKING)/rfval/200708.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200708

$(WORKING)/rfval/200711.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200711

$(WORKING)/rfval/200802.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200802

$(WORKING)/rfval/200805.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200805

$(WORKING)/rfval/200808.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200808

$(WORKING)/rfval/200811.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200811

$(WORKING)/rfval/200902.pickle: rfval.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfval.py 200902

# parcels-*
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
