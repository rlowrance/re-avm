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
# Dell has 8C, 16T
# HP210z has 4C, 8T, 16 GB 
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200512.pickle
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200601.pickle
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200602.pickle
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200603.pickle
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200604.pickle
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200605.pickle
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200606.pickle
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200607.pickle
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200608.pickle
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200609.pickle
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200610.pickle
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200611.pickle
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200612.pickle
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200701.pickle
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200702.pickle
VALAVM_SWPN_ALL_DELL   += $(WORKING)/valavm/swpn-all/swpn-all-200703.pickle
VALAVM_SWPN_ALL_ROY    += $(WORKING)/valavm/swpn-all/swpn-all-200704.pickle
VALAVM_SWPN_ALL_ROY    += $(WORKING)/valavm/swpn-all/swpn-all-200705.pickle
VALAVM_SWPN_ALL_ROY    += $(WORKING)/valavm/swpn-all/swpn-all-200706.pickle
VALAVM_SWPN_ALL_ROY    += $(WORKING)/valavm/swpn-all/swpn-all-200707.pickle
VALAVM_SWPN_ALL_ROY    += $(WORKING)/valavm/swpn-all/swpn-all-200708.pickle
VALAVM_SWPN_ALL_ROY    += $(WORKING)/valavm/swpn-all/swpn-all-200709.pickle
VALAVM_SWPN_ALL_ROY    += $(WORKING)/valavm/swpn-all/swpn-all-200710.pickle
VALAVM_SWPN_ALL_ROY    += $(WORKING)/valavm/swpn-all/swpn-all-200711.pickle
VALAVM_SWPN_ALL_ROY    += $(WORKING)/valavm/swpn-all/swpn-all-200712.pickle
VALAVM_SWPN_ALL_ROY    += $(WORKING)/valavm/swpn-all/swpn-all-200801.pickle
VALAVM_SWPN_ALL_ROY    += $(WORKING)/valavm/swpn-all/swpn-all-200802.pickle
VALAVM_SWPN_ALL_ROY    += $(WORKING)/valavm/swpn-all/swpn-all-200803.pickle
VALAVM_SWPN_ALL_HP     += $(WORKING)/valavm/swpn-all/swpn-all-200804.pickle
VALAVM_SWPN_ALL_HP     += $(WORKING)/valavm/swpn-all/swpn-all-200805.pickle
VALAVM_SWPN_ALL_HP     += $(WORKING)/valavm/swpn-all/swpn-all-200806.pickle
VALAVM_SWPN_ALL_HP     += $(WORKING)/valavm/swpn-all/swpn-all-200807.pickle
VALAVM_SWPN_ALL_JUDITH += $(WORKING)/valavm/swpn-all/swpn-all-200808.pickle
VALAVM_SWPN_ALL_JUDITH += $(WORKING)/valavm/swpn-all/swpn-all-200809.pickle
VALAVM_SWPN_ALL_JUDITH += $(WORKING)/valavm/swpn-all/swpn-all-200810.pickle
VALAVM_SWPN_ALL_JUDITH += $(WORKING)/valavm/swpn-all/swpn-all-200811.pickle
VALAVM_SWPN_ALL_JUDITH += $(WORKING)/valavm/swpn-all/swpn-all-200812.pickle
VALAVM_SWPN_ALL_JUDITH += $(WORKING)/valavm/swpn-all/swpn-all-200901.pickle
VALAVM_SWPN_ALL_JUDITH += $(WORKING)/valavm/swpn-all/swpn-all-200902.pickle
VALAVM_SWPN_ALL += $(VALAVM_SWPN_ALL_DELL)
VALAVM_SWPN_ALL += $(VALAVM_SWPN_ALL_ROY)
VALAVM_SWPN_ALL += $(VALAVM_SWPN_ALL_HP)
VALAVM_SWPN_ALL += $(VALAVM_SWPN_ALL_JUDITH)
ALL += $(VALAVM_SWPN_ALL)

VALAVM_SW_ALL_HP     += $(WORKING)/valavm/sw-all/sw-all-200512.pickle
VALAVM_SW_ALL_HP     += $(WORKING)/valavm/sw-all/sw-all-200601.pickle
VALAVM_SW_ALL_HP     += $(WORKING)/valavm/sw-all/sw-all-200602.pickle
VALAVM_SW_ALL_HP     += $(WORKING)/valavm/sw-all/sw-all-200603.pickle
VALAVM_SW_ALL_ROY    += $(WORKING)/valavm/sw-all/sw-all-200604.pickle
VALAVM_SW_ALL_ROY    += $(WORKING)/valavm/sw-all/sw-all-200605.pickle
VALAVM_SW_ALL_ROY    += $(WORKING)/valavm/sw-all/sw-all-200606.pickle
VALAVM_SW_ALL_ROY    += $(WORKING)/valavm/sw-all/sw-all-200607.pickle
VALAVM_SW_ALL_ROY    += $(WORKING)/valavm/sw-all/sw-all-200608.pickle
VALAVM_SW_ALL_ROY    += $(WORKING)/valavm/sw-all/sw-all-200609.pickle
VALAVM_SW_ALL_ROY    += $(WORKING)/valavm/sw-all/sw-all-200610.pickle
VALAVM_SW_ALL_ROY    += $(WORKING)/valavm/sw-all/sw-all-200611.pickle
VALAVM_SW_ALL_ROY    += $(WORKING)/valavm/sw-all/sw-all-200612.pickle
VALAVM_SW_ALL_ROY    += $(WORKING)/valavm/sw-all/sw-all-200701.pickle
VALAVM_SW_ALL_ROY    += $(WORKING)/valavm/sw-all/sw-all-200702.pickle
VALAVM_SW_ALL_ROY    += $(WORKING)/valavm/sw-all/sw-all-200703.pickle
VALAVM_SW_ALL_HP     += $(WORKING)/valavm/sw-all/sw-all-200704.pickle
VALAVM_SW_ALL_HP     += $(WORKING)/valavm/sw-all/sw-all-200705.pickle
VALAVM_SW_ALL_HP     += $(WORKING)/valavm/sw-all/sw-all-200706.pickle
VALAVM_SW_ALL_HP     += $(WORKING)/valavm/sw-all/sw-all-200707.pickle
VALAVM_SW_ALL_HP     += $(WORKING)/valavm/sw-all/sw-all-200708.pickle
VALAVM_SW_ALL_HP     += $(WORKING)/valavm/sw-all/sw-all-200709.pickle
VALAVM_SW_ALL_HP     += $(WORKING)/valavm/sw-all/sw-all-200710.pickle
VALAVM_SW_ALL_HP     += $(WORKING)/valavm/sw-all/sw-all-200711.pickle
VALAVM_SW_ALL_JUDITH += $(WORKING)/valavm/sw-all/sw-all-200712.pickle
VALAVM_SW_ALL_JUDITH += $(WORKING)/valavm/sw-all/sw-all-200801.pickle
VALAVM_SW_ALL_JUDITH += $(WORKING)/valavm/sw-all/sw-all-200801.pickle
VALAVM_SW_ALL_JUDITH += $(WORKING)/valavm/sw-all/sw-all-200803.pickle
VALAVM_SW_ALL_JUDITH += $(WORKING)/valavm/sw-all/sw-all-200804.pickle
VALAVM_SW_ALL_JUDITH += $(WORKING)/valavm/sw-all/sw-all-200805.pickle
VALAVM_SW_ALL_JUDITH += $(WORKING)/valavm/sw-all/sw-all-200806.pickle
VALAVM_SW_ALL_JUDITH += $(WORKING)/valavm/sw-all/sw-all-200807.pickle
VALAVM_SW_ALL_X      += $(WORKING)/valavm/sw-all/sw-all-200808.pickle
VALAVM_SW_ALL_X      += $(WORKING)/valavm/sw-all/sw-all-200809.pickle
VALAVM_SW_ALL_X      += $(WORKING)/valavm/sw-all/sw-all-200810.pickle
VALAVM_SW_ALL_X      += $(WORKING)/valavm/sw-all/sw-all-200811.pickle
VALAVM_SW_ALL_X      += $(WORKING)/valavm/sw-all/sw-all-200812.pickle
VALAVM_SW_ALL_X      += $(WORKING)/valavm/sw-all/sw-all-200901.pickle
VALAVM_SW_ALL_X      += $(WORKING)/valavm/sw-all/sw-all-200902.pickle
VALAVM_SW_ALL += $(VALAVM_SW_ALL_DELL)
VALAVM_SW_ALL += $(VALAVM_SW_ALL_ROY)
VALAVM_SW_ALL += $(VALAVM_SW_ALL_HP)
VALAVM_SW_ALL += $(VALAVM_SW_ALL_JUDITH)
VALAVM_SW_ALL += $(VALAVM_SW_ALL_X)
ALL += $(VALAVM_SW_ALL)

VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200512.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200601.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200602.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200603.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200604.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200605.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200606.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200607.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200608.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200609.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200610.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200611.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200612.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200701.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200702.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200703.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200704.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200705.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200706.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200707.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200708.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200709.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200710.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200711.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200712.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200801.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200802.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200803.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200804.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200805.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200806.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200807.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200808.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200809.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200810.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200811.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200812.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200901.pickle
VALAVM_SWPN_BEST1 += $(WORKING)/valavm/swpn-best1/swpn-best1-200902.pickle
#ALL += $(VALAVM_SWPN_BEST1)

# define the charts
# NOTE: many charts of historic interest only and were not used in the final report
#
# name one chart from each set of used chart
# Note: many charts supported preliminary analysis not in the final paper
CHARTSDIR = $(WORKING)/charts
CHARTS01 += $(CHARTSDIR)/01/median-price.pdf
CHARTS06 += $(CHARTSDIR)/06/a.pdf
CHARTS07 += $(CHARTSDIR)/07/a-nbest-1-nworst-0.txt
ALLCHARTS = $(CHARTS01) $(CHARTS02) $(CHARTS03)

ALL += $(ALLCHARTS)

# pick a representative of all the rank_models
ALL += $(WORKING)/rank_models/200512.pickle

.PHONY : all
all: $(ALL)

.PHONY : dell-swpn-all
dell-swpn-all: $(VALAVM_SWPN_ALL_DELL)

.PHONY : roy-sw-all
roy-sw-all: $(VALAVM_SW_ALL_ROY)

.PHONY : roy-swpn-all
roy-swpn-all: $(VALAVM_SWPN_ALL_ROY)

.PHONY : hp-sw-all
hp-sw-all: $(VALAVM_SW_ALL_HP)

.PHONY : hp-swpn-all
hp-swpn-all: $(VALAVM_SW_ALL_HP)

.PHONY : judith-swpn-all judith-sw-all
#$(info $(VALAVM_SWPN_ALL_JUDITH))
judith-swpn-all: $(VALAVM_SWPN_ALL_JUDITH)

judith-sw-all: $(VALAVM_SW_ALL_JUDITH)

.PHONY: charts01
charts01: $(CHARTS01)

.PHONY: chart06
charts06: $(CHARTS06)


# builds for charts actually used

# chart01
$(CHARTSDIR)/01/data.pickle: chart01.py $(WORKING)/samples-train.csv
	$(PYTHON) chart01.py --data

$(CHARTSDIR)/01/median-price.pdf: chart01.py $(CHARTSDIR)/01/data.pickle
	$(PYTHON) chart01.py
	
#$(WORKING)/chart01/data.pickle: chart01.py $(WORKING)/samples-train-validate.csv
#	$(PYTHON) chart01.py --data
#
#$(WORKING)/chart01/median-price.pdf: chart01.py $(WORKING)/chart01/data.pickle
#	$(PYTHON) chart01.py
	
# chart06 
$(CHARTSDIR)/06/data.pickle: chart06.py $(CHARTSDIR)/01/data.pickle $(VALAVM)
	$(PYTHON) chart06.py  --data

$(CHARTSDIR)/06/a.pdf: chart06.py $(CHARTSDIR)/06/data.pickle
	$(PYTHON) chart06.py 

# chart07
$(WORKING)/chart07/data.pickle: chart07.py $(VALAVM_FITTED)
	$(PYTHON) chart07.py --data

$(WORKING)/chart07/a-nbest-1-nworst-0.txt: chart07.py $(WORKING)/chart07/data.pickle
	$(PYTHON) chart07.py

.PHONY : parcels-features
parcels-features: $(WORKING)/parcels-features-census_tract.csv $(WORKING)/parcels-features-zip5.csv


# rank_models
$(WORKING)/rank_models/200512.pickle: $(WORKING)/chart06/data.pickle
	$(PYTHON) rank_models.py


$(WORKING)/census-features-derived.csv: census-features.py layout_census.py
	$(PYTHON) census-features.py

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

# valavm
valavm_dep += valavm.py
valavm_dep += AVM.py
valavm_dep += AVM_gradient_boosting_regressor.py
valavm_dep += AVM_random_forest_regressor.py
valavm_dep += AVM_elastic_net.py
valavm_dep += $(WORKING)/samples-train.csv

# valavm

$(WORKING)/valavm/sw-all/%.pickle: $(valavm_dep)
	$(PYTHON) valavm.py $*

$(WORKING)/valavm/swpn-all/%.pickle: $(valavm_dep)
	$(PYTHON) valavm.py $*

# valavm-fitted

valavm_fitted_dep += $(valavm_dep)
valavm_fitted_dep += $(WORKING)/rank_models/200512.pickle

$(WORKING)/valavm/fitted-%.pickle: $(valavm_fitted_dep)
	$(PYTHON) valavm.py $* --onlyfitted 1


