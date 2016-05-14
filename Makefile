# --debug=basic
# disable built-in rules
.SUFFIXES:



PYTHON = ~/anaconda/bin/python

WORKING = ../data/working

ALL += $(WORKING)/census-features-derived.csv

# CHART02 and RFBOUND are obsoleted by RFVAL
# their rules and recipes are in rfbound.mk

# Adjust the VARIABLES below to rebuild optimally on available systems
# DON'T OVERSCHEDULE THE CPUs and RAM, or the system will start to swap
# Carmen  Judith's MacPro has 4C, 8T, 64 GB
# Elektra Roy's MacPro has 6C, 12T, 64 GB
# HP210z has 4C, 8T, 16 GB 
VALAVM_A += $(WORKING)/valavm/200512.pickle
VALAVM_A += $(WORKING)/valavm/200601.pickle
VALAVM_A += $(WORKING)/valavm/200602.pickle
VALAVM_A += $(WORKING)/valavm/200603.pickle
VALAVM_A += $(WORKING)/valavm/200604.pickle
VALAVM_A += $(WORKING)/valavm/200605.pickle
VALAVM_A += $(WORKING)/valavm/200606.pickle
VALAVM_A += $(WORKING)/valavm/200607.pickle
VALAVM_A += $(WORKING)/valavm/200608.pickle
VALAVM_A += $(WORKING)/valavm/200609.pickle
VALAVM_A += $(WORKING)/valavm/200610.pickle
VALAVM_A += $(WORKING)/valavm/200611.pickle
VALAVM_A += $(WORKING)/valavm/200612.pickle
VALAVM_A += $(WORKING)/valavm/200701.pickle
VALAVM_A += $(WORKING)/valavm/200702.pickle
VALAVM_A += $(WORKING)/valavm/200703.pickle
VALAVM_A += $(WORKING)/valavm/200704.pickle
VALAVM_A += $(WORKING)/valavm/200705.pickle
VALAVM_A += $(WORKING)/valavm/200706.pickle
VALAVM_A += $(WORKING)/valavm/200707.pickle
VALAVM_A += $(WORKING)/valavm/200708.pickle
VALAVM_A += $(WORKING)/valavm/200709.pickle
VALAVM_A += $(WORKING)/valavm/200710.pickle
VALAVM_A += $(WORKING)/valavm/200711.pickle
VALAVM_B += $(WORKING)/valavm/200712.pickle
VALAVM_B += $(WORKING)/valavm/200801.pickle
VALAVM_B += $(WORKING)/valavm/200802.pickle
VALAVM_B += $(WORKING)/valavm/200803.pickle
VALAVM_B += $(WORKING)/valavm/200804.pickle
VALAVM_B += $(WORKING)/valavm/200805.pickle
VALAVM_B += $(WORKING)/valavm/200806.pickle
VALAVM_B += $(WORKING)/valavm/200807.pickle
VALAVM_B += $(WORKING)/valavm/200808.pickle
VALAVM_B += $(WORKING)/valavm/200809.pickle
VALAVM_B += $(WORKING)/valavm/200810.pickle
VALAVM_B += $(WORKING)/valavm/200811.pickle
VALAVM_B += $(WORKING)/valavm/200812.pickle
VALAVM_B += $(WORKING)/valavm/200901.pickle
VALAVM_B += $(WORKING)/valavm/200902.pickle
VALAVM += $(VALAVM_A) $(VALAVM_B)
ALL += $(VALAVM)

include charts.makefile  # yields variable CHARTS
ALL += $(CHARTS)

.PHONY : all
all: $(ALL)

# builds for VALAVM on separate systems
# invocations
#   make valavm_A
#   make valavm_B
.PHONY : valavm_A
valavm_A: $(VALAVM_A)

.PHONY : valavm_B
valavm_B: $(VALAVM_B)

.PHONY : parcels-features
parcels-features: $(WORKING)/parcels-features-census_tract.csv $(WORKING)/parcels-features-zip5.csv


$(WORKING)/census-features-derived.csv: census-features.py layout_census.py
	$(PYTHON) census-features.py

# valavm
valavm_dep += valavm.py
valavm_dep += AVM.py
valavm_dep += AVM_gradient_boosting_regressor.py
valavm_dep += AVM_random_forest_regressor.py
valavm_dep += AVM_elastic_net.py
valavm_dep += $(WORKING)/samples-train.csv

# valavm
$(WORKING)/valavm/200612.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200612

$(WORKING)/valavm/200701.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200701

$(WORKING)/valavm/200702.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200702 

$(WORKING)/valavm/200703.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200703 

$(WORKING)/valavm/200704.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200704 

$(WORKING)/valavm/200705.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200705 

$(WORKING)/valavm/200706.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200706 

$(WORKING)/valavm/200707.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200707 

$(WORKING)/valavm/200708.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200708 

$(WORKING)/valavm/200709.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200709 

$(WORKING)/valavm/200710.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200710 

$(WORKING)/valavm/200711.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200711 

$(WORKING)/valavm/200712.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200712 

$(WORKING)/valavm/200801.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200801 

$(WORKING)/valavm/200802.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200802 

$(WORKING)/valavm/200803.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200803 

$(WORKING)/valavm/200804.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200804 

$(WORKING)/valavm/200805.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200805 


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
