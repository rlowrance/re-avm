# --debug=basic
# disable built-in rules
.SUFFIXES:


PYTHON = ~/anaconda2/bin/python

WORKING = ../data/working

ALL += $(WORKING)/census-features-derived.csv

#ALL += valavm.makefile

# define the charts
# NOTE: many charts of historic interest only and were not used in the final report
#
# name one chart from each set of used chart
# Note: many charts supported preliminary analysis not in the final paper
CHART01 += $(WORKING)/chart01/median-price.pdf

CHART06 += $(WORKING)/chart06/s-all-global/a.pdf
CHART06 += $(WORKING)/chart06/sw-all-global/a.pdf
CHART06 += $(WORKING)/chart06/swp-all-global/a.pdf
CHART06 += $(WORKING)/chart06/swpn-all-global/a.pdf

CHART07 += $(WORKING)/chart07/s-all-global/b.txt
CHART07 += $(WORKING)/chart07/sw-all-global/b.txt
CHART07 += $(WORKING)/chart07/swp-all-global/b.txt
CHART07 += $(WORKING)/chart07/swpn-all-global/b.txt

CHART08 += $(WORKING)/chart08/a.txt

ALLCHARTS = $(CHART01) $(CHART06) $(CHART07) $(CHART08)

ALL += $(ALLCHARTS)

# pick a representative of all the rank_models
#ALL += $(WORKING)/rank_models/200512.pickle

.PHONY : all
all: $(ALL)

.PHONY: chart01
chart01: $(CHART01)

.PHONY: chart06
chart06: $(CHART06)

.PHONY: chart07
chart07: $(CHART07)

.PHONY: chart08
chart08: $(CHART08)

.PHONY: charts
charts: chart01 chart06 chart07 chart08

# census-features-derived.csv
$(WORKING)/census-features-derived.csv: census-features.py layout_census.py
	$(PYTHON) census-features.py


# builds for charts actually used
# NOTE: some charts were created and not used ihe final report

# chart01
$(WORKING)/chart01/0data.pickle: chart01.py $(WORKING)/samples-train.csv
	$(PYTHON) chart01.py --data

$(WORKING)/chart01/median-price.pdf: chart01.py $(WORKING)/chart01/0data.pickle
	$(PYTHON) chart01.py
	
# chart06 
$(WORKING)/chart06/%/0data.pickle: chart06.py $(WORKING)/chart01/0data.pickle $(VALAVM)
	$(PYTHON) chart06.py $* --data

$(WORKING)/chart06/%/a.pdf: chart06.py $(WORKING)/chart06/%/0data.pickle
	$(PYTHON) chart06.py $* 

.PRECIOUS: $(WORKING)/chart06/%/0data.pickle  # otherwise make deletes intermediate files

# chart07
$(WORKING)/chart07/%/0data.pickle: chart07.py $(WORKING)/valavm/%/200512.pickle
	$(PYTHON) chart07.py $* --data

$(WORKING)/chart07/%/b.txt: chart07.py $(WORKING)/chart07/%/0data.pickle
	$(PYTHON) chart07.py $* 

.PRECIOUS: $(WORKING)/chart07/%/0data.pickle  # otherwise make deletes intermediate files

# OLD chart07
#$(WORKING)/chart07/s-all/0data.pickle: chart07.py $(VALAVM_FITTED)
#	$(PYTHON) chart07.py s-all --data
#
#$(WORKING)/chart07/s-all/b.txt: chart07.py $(WORKING)/chart07/s-all/0data.pickle
#	$(PYTHON) chart07.py s-all
#
#$(WORKING)/chart07/sw-all/0data.pickle: chart07.py $(VALAVM_FITTED)
#	$(PYTHON) chart07.py sw-all --data
#
#$(WORKING)/chart07/sw-all/b.txt: chart07.py $(WORKING)/chart07/sw-all/0data.pickle
#	$(PYTHON) chart07.py sw-all
#
#$(WORKING)/chart07/swp-all/0data.pickle: chart07.py $(VALAVM_FITTED)
#	$(PYTHON) chart07.py swp-all --data
#
#$(WORKING)/chart07/swp-all/b.txt: chart07.py $(WORKING)/chart07/swp-all/0data.pickle
#	$(PYTHON) chart07.py swp-all
#
#$(WORKING)/chart07/swpn-all/0data.pickle: chart07.py $(VALAVM_FITTED)
#	$(PYTHON) chart07.py swpn-all --data
#
#$(WORKING)/chart07/swpn-all/b.txt: chart07.py $(WORKING)/chart07/swpn-all/0data.pickle
#	$(PYTHON) chart07.py swpn-all
#
# chart08
chart08deps += $(WORKING)/chart07/s-all-global/0data.pickle
chart08deps += $(WORKING)/chart07/sw-all-global/0data.pickle
chart08deps += $(WORKING)/chart07/swp-all-global/0data.pickle
chart08deps += $(WORKING)/chart07/swpn-all-global/0data.pickle

$(WORKING)/chart08/0data.pickle: chart08.py $(chart08deps)
	$(PYTHON) chart08.py --data

$(WORKING)/chart08/a.txt: chart08.py $(WORKING)/chart08/0data.pickle
	$(PYTHON) chart08.py 

.PHONY : parcels-features
parcels-features: $(WORKING)/parcels-features-census_tract.csv $(WORKING)/parcels-features-zip5.csv


# rank_models
$(WORKING)/rank_models/200512.pickle: rank_models.py $(WORKING)/chart06/data.pickle
	$(PYTHON) rank_models.py


# parcels-*
$(WORKING)/parcels-features-census_tract.csv: parcels-features.py layout_parcels.py
	$(PYTHON) parcels-features.py --geo census_tract

$(WORKING)/parcels-features-zip5.csv: parcels-features.py layout_parcels.py
	$(PYTHON) parcels-features.py --geo zip5

# transactions-al-g-srf
RLA = $(WORKING)/real-estate-log-angeles
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-deeds-090402_07/CAC06037F1.zip
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-deeds-090402_07/CAC06037F2.zip
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-deeds-090402_07/CAC06037F3.zip
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-deeds-090402_07/CAC06037F4.zip
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-deeds-090402_09/CAC06037F5.zip
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-deeds-090402_09/CAC06037F6.zip
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-deeds-090402_09/CAC06037F7.zip
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-deeds-090402_09/CAC06037F8.zip
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-taxrolls-090402_05/CAC06037F1.zip
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-taxrolls-090402_05/CAC06037F2.zip
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-taxrolls-090402_05/CAC06037F3.zip
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-taxrolls-090402_05/CAC06037F4.zip
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-taxrolls-090402_05/CAC06037F5.zip
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-taxrolls-090402_05/CAC06037F6.zip
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-taxrolls-090402_05/CAC06037F7.zip
TRANSACTIONS-AL-G-SFR-DEP += $(RLA)/corelogic-taxrolls-090402_05/CAC06037F8.zip
TRANSACTIONS-AL-G-SFR-DEP += $(WORKING)/parcels-features-census_tract.csv
TRANSACTIONS-AL-G-SFR-DEP += $(WORKING)/parcels-features-zip5.csv
$(WORKING)/transactions-al-g-sfr.csv: transactions.py $(TRANSACTIONS-AL-G-SFR-DEP)
	$(PYTHON) transactions.py

# samples.testcsv
#$(WORKING)/samples-test%csv $(WORKING)/samples-train%csv $(WORKING)/samples-train-validate%csv $(WORKING)/samples-validate%csv: samples.py $(WORKING)/transactions-al-g-sfr.csv
#	$(PYTHON) samples.py

# summarize-samples-train.csv
#$(WORKING)/summarize-samples-train.csv: summarize-df.py summarize.py
#	$(PYTHON) summarize-df.py --in samples-train.csv 

# valavm
valavm_dep += valavm.py
valavm_dep += AVM.py
valavm_dep += AVM_gradient_boosting_regressor.py
valavm_dep += AVM_random_forest_regressor.py
valavm_dep += AVM_elastic_net.py
valavm_dep += $(WORKING)/samples-train.csv

# valavm

# define targets $(WORKING)/valavm/{feature_group}-all-{locality}-{system}
# so that these invocations of make will work
#   make -j N {feature_group}-all-{locality}-{system}
# where N is 
#  16 on {system} = dell
#   4 on {system} = hp
#   7 on {system} = judith
#   8 on {system} = roy

#include valavm.makefile

valavm.makefile: valavm.py
	$(PYTHON) valavm.py --makefile

$(WORKING)/valavm/s-all/%.pickle: $(valavm_dep)
	$(PYTHON) valavm.py $*

$(WORKING)/valavm/sw-all/%.pickle: $(valavm_dep)
	$(PYTHON) valavm.py $*

$(WORKING)/valavm/swp-all/%.pickle: $(valavm_dep)
	$(PYTHON) valavm.py $*

$(WORKING)/valavm/swpn-all/%.pickle: $(valavm_dep)
	$(PYTHON) valavm.py $*

# valavm-fitted

valavm_fitted_dep += $(valavm_dep)
valavm_fitted_dep += $(WORKING)/rank_models/200512.pickle

$(WORKING)/valavm/fitted-%.pickle: $(valavm_fitted_dep)
	$(PYTHON) valavm.py $* --onlyfitted 1


