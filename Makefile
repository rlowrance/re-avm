# --debug=basic
# disable built-in rules
.SUFFIXES:



PYTHON = ~/anaconda/bin/python

WORKING = ../data/working

ALL += $(WORKING)/census-features-derived.csv
ALL += $(WORKING)/chart-01.txt

ALL += $(WORKING)/chart-02-max_depth-2004.pdf  # representative of many 
ALL += $(WORKING)/chart-02-max_features-2004.pdf  # representative of many 

MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200902-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200811-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200808-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200805-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200802-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200711-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200708-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200705-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200702-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200611-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200608-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200605-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200602-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200611-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200608-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200605-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200602-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200511-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200508-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200505-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200502-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200411-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200408-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200405-folds-10.pickle
MAX_DEPTH += $(WORKING)/ege-rfbound-max_depth-200402-folds-10.pickle
ALL += $(MAX_DEPTH)

MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200402-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200405-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200408-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200411-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200502-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200505-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200508-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200511-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200602-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200605-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200608-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200611-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200702-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200705-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200708-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200711-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200802-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200805-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200808-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200811-folds-10.pickle
MAX_FEATURES += $(WORKING)/ege-rfbound-max_features-200902-folds-10.pickle
ALL += $(MAX_FEATURES)

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

# chart-NN
$(WORKING)/chart-01.txt: chart-01.py $(WORKING)/chart-01.data.pickle
	$(PYTHON) chart-01.py
	
$(WORKING)/chart-01.data.pickle: chart-01.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) chart-01.py --data

$(WORKING)/chart-02-max_depth-2004.pdf: chart-02.py $(WORKING)/chart-02-max_depth.data.pickle
	$(PYTHON) chart-02.py max_depth

$(WORKING)/chart-02-max_depth.data.pickle: chart-02.py  $(MAX_DEPTH)
	$(PYTHON) chart-02.py max_depth --data

$(WORKING)/chart-02-max_features-2004.pdf: chart-02.py $(WORKING)/chart-02-max_features.data.pickle
	$(PYTHON) chart-02.py max_features

$(WORKING)/chart-02-max_features.data.pickle: chart-02.py $(MAX_FEATURES)
	$(PYTHON) chart-02.py max_features --data

# ege-rbbound-max_depth-*-folds-10.pickle
$(WORKING)/ege-rfbound-max_depth-200402-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200402 --folds 10

$(WORKING)/ege-rfbound-max_depth-200405-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200405 --folds 10

$(WORKING)/ege-rfbound-max_depth-200408-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200408 --folds 10

$(WORKING)/ege-rfbound-max_depth-200411-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200411 --folds 10

$(WORKING)/ege-rfbound-max_depth-200502-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200502 --folds 10

$(WORKING)/ege-rfbound-max_depth-200505-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200505 --folds 10

$(WORKING)/ege-rfbound-max_depth-200508-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200508 --folds 10

$(WORKING)/ege-rfbound-max_depth-200511-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200511 --folds 10

$(WORKING)/ege-rfbound-max_depth-200602-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200602 --folds 10

$(WORKING)/ege-rfbound-max_depth-200605-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200605 --folds 10

$(WORKING)/ege-rfbound-max_depth-200608-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200608 --folds 10

$(WORKING)/ege-rfbound-max_depth-200611-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200611 --folds 10

$(WORKING)/ege-rfbound-max_depth-200702-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200702 --folds 10

$(WORKING)/ege-rfbound-max_depth-200705-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200705 --folds 10

$(WORKING)/ege-rfbound-max_depth-200708-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200708 --folds 10

$(WORKING)/ege-rfbound-max_depth-200711-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200711 --folds 10

$(WORKING)/ege-rfbound-max_depth-200802-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200802 --folds 10

$(WORKING)/ege-rfbound-max_depth-200805-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200805 --folds 10

$(WORKING)/ege-rfbound-max_depth-200808-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200808 --folds 10

$(WORKING)/ege-rfbound-max_depth-200811-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200811 --folds 10

$(WORKING)/ege-rfbound-max_depth-200902-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_depth 200902 --folds 10

# ege-rbbound-max_features-*-folds-10.pickle
$(WORKING)/ege-rfbound-max_features-200402-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200402 --folds 10

$(WORKING)/ege-rfbound-max_features-200405-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200405 --folds 10

$(WORKING)/ege-rfbound-max_features-200408-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200408 --folds 10

$(WORKING)/ege-rfbound-max_features-200411-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200411 --folds 10

$(WORKING)/ege-rfbound-max_features-200502-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200502 --folds 10

$(WORKING)/ege-rfbound-max_features-200505-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200505 --folds 10

$(WORKING)/ege-rfbound-max_features-200508-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200508 --folds 10

$(WORKING)/ege-rfbound-max_features-200511-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200511 --folds 10

$(WORKING)/ege-rfbound-max_features-200602-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200602 --folds 10

$(WORKING)/ege-rfbound-max_features-200605-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200605 --folds 10

$(WORKING)/ege-rfbound-max_features-200608-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200608 --folds 10

$(WORKING)/ege-rfbound-max_features-200611-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200611 --folds 10

$(WORKING)/ege-rfbound-max_features-200702-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200702 --folds 10

$(WORKING)/ege-rfbound-max_features-200705-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200705 --folds 10

$(WORKING)/ege-rfbound-max_features-200708-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200708 --folds 10

$(WORKING)/ege-rfbound-max_features-200711-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200711 --folds 10

$(WORKING)/ege-rfbound-max_features-200802-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200802 --folds 10

$(WORKING)/ege-rfbound-max_features-200805-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200805 --folds 10

$(WORKING)/ege-rfbound-max_features-200808-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200808 --folds 10

$(WORKING)/ege-rfbound-max_features-200811-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200811 --folds 10

$(WORKING)/ege-rfbound-max_features-200902-folds-10.pickle: ege.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) ege.py --rfbound max_features 200902 --folds 10

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
