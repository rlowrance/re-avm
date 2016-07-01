# rules and recipes for rfbound, which is obsolete (use rfval instead)
# chart-02 creates output using the result of rfbound, so its included here
# --debug=basic
# disable built-in rules
.SUFFIXES:



PYTHON = ~/anaconda/bin/python

WORKING = ../data/working

CHART02 += $(WORKING)/chart-02-max_depth-2004.pdf  # representative of many 
CHART02 += $(WORKING)/chart-02-max_features-2004.pdf  # representative of many 

MAX_DEPTH += $(WORKING)/rfbound/max_depth-200902-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200811-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200808-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200805-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200802-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200711-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200708-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200705-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200702-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200611-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200608-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200605-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200602-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200611-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200608-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200605-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200602-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200511-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200508-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200505-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200502-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200411-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200408-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200405-10.pickle
MAX_DEPTH += $(WORKING)/rfbound/max_depth-200402-10.pickle

MAX_FEATURES += $(WORKING)/rfbound/max_features-200402-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200405-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200408-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200411-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200502-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200505-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200508-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200511-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200602-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200605-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200608-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200611-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200702-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200705-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200708-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200711-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200802-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200805-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200808-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200811-10.pickle
MAX_FEATURES += $(WORKING)/rfbound/max_features-200902-10.pickle

RFBOUND += $(MAX_DEPTH) $(MAX_FEATURES)

.PHONY : all
all: $(ALL)


# chart-02
$(WORKING)/chart-02-max_depth-2004.pdf: chart-02.py $(WORKING)/chart-02-max_depth.data.pickle
	$(PYTHON) chart-02.py max_depth

$(WORKING)/chart-02-max_depth.data.pickle: chart-02.py  $(MAX_DEPTH)
	$(PYTHON) chart-02.py max_depth --data

$(WORKING)/chart-02-max_features-2004.pdf: chart-02.py $(WORKING)/chart-02-max_features.data.pickle
	$(PYTHON) chart-02.py max_features

$(WORKING)/chart-02-max_features.data.pickle: chart-02.py $(MAX_FEATURES)
	$(PYTHON) chart-02.py max_features --data


# rbbound-max_depth-*-folds-10.pickle
$(WORKING)/rfbound/max_depth-200402-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200402 10

$(WORKING)/rfbound/max_depth-200405-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200405  10

$(WORKING)/rfbound/max_depth-200408-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200408  10

$(WORKING)/rfbound/max_depth-200411-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200411  10

$(WORKING)/rfbound/max_depth-200502-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200502  10

$(WORKING)/rfbound/max_depth-200505-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200505  10

$(WORKING)/rfbound/max_depth-200508-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200508  10

$(WORKING)/rfbound/max_depth-200511-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200511  10

$(WORKING)/rfbound/max_depth-200602-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200602  10

$(WORKING)/rfbound/max_depth-200605-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200605  10

$(WORKING)/rfbound/max_depth-200608-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200608  10

$(WORKING)/rfbound/max_depth-200611-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200611  10

$(WORKING)/rfbound/max_depth-200702-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200702  10

$(WORKING)/rfbound/max_depth-200705-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200705  10

$(WORKING)/rfbound/max_depth-200708-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200708  10

$(WORKING)/rfbound/max_depth-200711-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200711  10

$(WORKING)/rfbound/max_depth-200802-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200802  10

$(WORKING)/rfbound/max_depth-200805-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200805  10

$(WORKING)/rfbound/max_depth-200808-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200808  10

$(WORKING)/rfbound/max_depth-200811-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200811  10

$(WORKING)/rfbound/max_depth-200902-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_depth  200902  10

# rfbound-max_features-*-10.pickle
$(WORKING)/rfbound/max_features-200402-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200402  10

$(WORKING)/rfbound/max_features-200405-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200405  10

$(WORKING)/rfbound/max_features-200408-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200408  10

$(WORKING)/rfbound/max_features-200411-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200411  10

$(WORKING)/rfbound/max_features-200502-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200502  10

$(WORKING)/rfbound/max_features-200505-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200505  10

$(WORKING)/rfbound/max_features-200508-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200508  10

$(WORKING)/rfbound/max_features-200511-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200511  10

$(WORKING)/rfbound/max_features-200602-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200602  10

$(WORKING)/rfbound/max_features-200605-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200605  10

$(WORKING)/rfbound/max_features-200608-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200608  10

$(WORKING)/rfbound/max_features-200611-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200611  10

$(WORKING)/rfbound/max_features-200702-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200702  10

$(WORKING)/rfbound/max_features-200705-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200705  10

$(WORKING)/rfbound/max_features-200708-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200708  10

$(WORKING)/rfbound/max_features-200711-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200711  10

$(WORKING)/rfbound/max_features-200802-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200802  10

$(WORKING)/rfbound/max_features-200805-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200805  10

$(WORKING)/rfbound/max_features-200808-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200808  10

$(WORKING)/rfbound/max_features-200811-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200811  10

$(WORKING)/rfbound/max_features-200902-10.pickle: rfbound.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) rfbound.py max_features  200902  10
