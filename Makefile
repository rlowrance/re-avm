# --debug=basic
# disable built-in rules
.SUFFIXES:


PYTHON = ~/anaconda2/bin/python

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

VALAVM_FITTED += $(WORKING)/valavm/200512-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200601-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200602-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200603-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200604-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200605-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200606-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200607-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200608-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200609-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200610-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200611-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200612-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200701-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200702-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200703-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200704-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200705-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200706-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200707-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200708-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200709-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200710-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200711-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200712-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200801-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200802-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200803-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200804-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200805-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200806-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200807-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200808-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200809-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200810-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200811-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200812-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200901-fitted.pickle
VALAVM_FITTED += $(WORKING)/valavm/200902-fitted.pickle
ALL += $(VALAVM_FITTED)

# define the charts
# NOTE: many charts of historic interest only and were not used in the final report
#
# name one chart from each set of used chart
# Note: many charts supported preliminary analysis not in the final paper
CHARTS += $(WORKING)/chart01/median-price.pdf
CHARTS += $(WORKING)/chart06/a.pdf

ALL += $(CHARTS)

.PHONY : all
all: $(ALL)

# builds for charts actually used

# chart01
$(WORKING)/chart01/data.pickle: chart01.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) chart01.py --data

$(WORKING)/chart01/median-price.pdf: chart01.py $(WORKING)/chart01/data.pickle
	$(PYTHON) chart01.py
	
# chart06 
$(WORKING)/chart06/data.pickle: chart06.py $(WORKING)/chart01/data.pickle $(VALAVM)
	$(PYTHON) chart06.py  --data

$(WORKING)/chart06/a.pdf: chart06.py $(WORKING)/chart06/data.pickle
	$(PYTHON) chart06.py 

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

$(WORKING)/valavm/%-pickle: $(valavm_dep)
	$(PYTHON) valavm.py $*

# valavm-fitted

valavm_fitted_deps += $(valavm_dep)
valavm_fitted_deps += $(WORKING)/best-models.pickle

$(WORKING)/valavm/%-fitted.pickle: $(valavm_dep)
	$(PYTHON) valavm.py $* --grid $(WORKING)/best-models.pickle 


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
