# --debug=basic
# disable built-in rules
.SUFFIXES:



PYTHON = ~/anaconda/bin/python

WORKING = ../data/working

ALL += $(WORKING)/census-features-derived.csv

# CHART02 and RFBOUND are obsoleted by RFVAL
# their rules and recipes are in rfbound.mk

# make -j 12 runs OK to make all the valavm objects
VALAVM_ANIL += $(WORKING)/valavm-anil/200612.pickle
VALAVM_ANIL += $(WORKING)/valavm-anil/200701.pickle
VALAVM_ANIL += $(WORKING)/valavm-anil/200702.pickle
VALAVM_ANIL += $(WORKING)/valavm-anil/200703.pickle
VALAVM_ANIL += $(WORKING)/valavm-anil/200704.pickle
VALAVM_ANIL += $(WORKING)/valavm-anil/200705.pickle
VALAVM_ANIL += $(WORKING)/valavm-anil/200706.pickle
VALAVM_ANIL += $(WORKING)/valavm-anil/200707.pickle
VALAVM_ANIL += $(WORKING)/valavm-anil/200708.pickle
VALAVM_ANIL += $(WORKING)/valavm-anil/200709.pickle
VALAVM_ANIL += $(WORKING)/valavm-anil/200710.pickle
VALAVM_ANIL += $(WORKING)/valavm-anil/200711.pickle
VALAVM_ANIL += $(WORKING)/valavm-anil/200712.pickle
VALAVM_ROY += $(WORKING)/valavm-roy/200612.pickle
VALAVM_ROY += $(WORKING)/valavm-roy/200701.pickle
VALAVM_ROY += $(WORKING)/valavm-roy/200702.pickle
VALAVM_ROY += $(WORKING)/valavm-roy/200703.pickle
VALAVM_ROY += $(WORKING)/valavm-roy/200704.pickle
VALAVM_ROY += $(WORKING)/valavm-roy/200705.pickle
VALAVM_ROY += $(WORKING)/valavm-roy/200706.pickle
VALAVM_ROY += $(WORKING)/valavm-roy/200707.pickle
VALAVM_ROY += $(WORKING)/valavm-roy/200708.pickle
VALAVM_ROY += $(WORKING)/valavm-roy/200709.pickle
VALAVM_ROY += $(WORKING)/valavm-roy/200710.pickle
VALAVM_ROY += $(WORKING)/valavm-roy/200711.pickle
VALAVM_ROY += $(WORKING)/valavm-roy/200712.pickle
ALL += $(VALAVM_ANIL) $(VALAVM_ROY)

VALGBR += $(WORKING)/valgbr/200402.pickle
VALGBR += $(WORKING)/valgbr/200405.pickle
VALGBR += $(WORKING)/valgbr/200408.pickle
VALGBR += $(WORKING)/valgbr/200411.pickle
VALGBR += $(WORKING)/valgbr/200502.pickle
VALGBR += $(WORKING)/valgbr/200505.pickle
VALGBR += $(WORKING)/valgbr/200508.pickle
VALGBR += $(WORKING)/valgbr/200511.pickle
VALGBR += $(WORKING)/valgbr/200602.pickle
VALGBR += $(WORKING)/valgbr/200605.pickle
VALGBR += $(WORKING)/valgbr/200608.pickle
VALGBR += $(WORKING)/valgbr/200611.pickle
VALGBR += $(WORKING)/valgbr/200702.pickle
VALGBR += $(WORKING)/valgbr/200705.pickle
VALGBR += $(WORKING)/valgbr/200708.pickle
VALGBR += $(WORKING)/valgbr/200711.pickle
VALGBR += $(WORKING)/valgbr/200802.pickle
VALGBR += $(WORKING)/valgbr/200805.pickle
VALGBR += $(WORKING)/valgbr/200808.pickle
VALGBR += $(WORKING)/valgbr/200811.pickle
VALGBR += $(WORKING)/valgbr/200902.pickle
#ALL += $(VALGBR)

VALLIN += $(WORKING)/vallin/200402.pickle
VALLIN += $(WORKING)/vallin/200405.pickle
VALLIN += $(WORKING)/vallin/200408.pickle
VALLIN += $(WORKING)/vallin/200411.pickle
VALLIN += $(WORKING)/vallin/200502.pickle
VALLIN += $(WORKING)/vallin/200505.pickle
VALLIN += $(WORKING)/vallin/200508.pickle
VALLIN += $(WORKING)/vallin/200511.pickle
VALLIN += $(WORKING)/vallin/200602.pickle
VALLIN += $(WORKING)/vallin/200605.pickle
VALLIN += $(WORKING)/vallin/200608.pickle
VALLIN += $(WORKING)/vallin/200611.pickle
VALLIN += $(WORKING)/vallin/200702.pickle
VALLIN += $(WORKING)/vallin/200705.pickle
VALLIN += $(WORKING)/vallin/200708.pickle
VALLIN += $(WORKING)/vallin/200711.pickle
VALLIN += $(WORKING)/vallin/200802.pickle
VALLIN += $(WORKING)/vallin/200805.pickle
VALLIN += $(WORKING)/vallin/200808.pickle
VALLIN += $(WORKING)/vallin/200811.pickle
VALLIN += $(WORKING)/vallin/200902.pickle
#ALL += $(VALLIN)

VALRF += $(WORKING)/valrf/200402.pickle
VALRF += $(WORKING)/valrf/200405.pickle
VALRF += $(WORKING)/valrf/200408.pickle
VALRF += $(WORKING)/valrf/200411.pickle
VALRF += $(WORKING)/valrf/200502.pickle
VALRF += $(WORKING)/valrf/200505.pickle
VALRF += $(WORKING)/valrf/200508.pickle
VALRF += $(WORKING)/valrf/200511.pickle
VALRF += $(WORKING)/valrf/200602.pickle
VALRF += $(WORKING)/valrf/200605.pickle
VALRF += $(WORKING)/valrf/200608.pickle
VALRF += $(WORKING)/valrf/200611.pickle
VALRF += $(WORKING)/valrf/200702.pickle
VALRF += $(WORKING)/valrf/200705.pickle
VALRF += $(WORKING)/valrf/200708.pickle
VALRF += $(WORKING)/valrf/200711.pickle
VALRF += $(WORKING)/valrf/200802.pickle
VALRF += $(WORKING)/valrf/200805.pickle
VALRF += $(WORKING)/valrf/200808.pickle
VALRF += $(WORKING)/valrf/200811.pickle
VALRF += $(WORKING)/valrf/200902.pickle
#ALL += $(VALRF)

CHARTS += $(WORKING)/chart-01/median-price.pdf
# use max_depth as a proxy for both max_depth and max_features
# use 2004-02 as a proxy for all years YYYY and months MM
CHARTS += $(WORKING)/chart-03/max_depth-2004-02.pdf
CHARTS += $(WORKING)/chart-04/2004-02.pdf
#CHARTS += $(WORKING)/chart-05/2004.pdf
CHARTS += $(WORKING)/chart-06-anil/a.pdf  # plus others
CHARTS += $(WORKING)/chart-06-roy/a.pdf  # plus others

ALL += $(CHARTS)

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

.PHONY : chart-04
chart-04: $(WORKING)/chart-04/2004-02.pdf

.PHONY : chart-05
chart-05: $(WORKING)/chart-05/2004.pdf

$(WORKING)/census-features-derived.csv: census-features.py layout_census.py
	$(PYTHON) census-features.py

# chart-01
$(WORKING)/chart-01/median-price.pdf: chart-01.py $(WORKING)/chart-01/data.pickle
	$(PYTHON) chart-01.py
	
$(WORKING)/chart-01/data.pickle: chart-01.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) chart-01.py --data

# chart-03
#    max_depth is a proxy for both max_depth and max_features
#    2004-02 is a proxy for all years YYYY and all months MM
CHART03REDUCTION = $(WORKING)/chart-03/data.pickle

$(CHART03REDUCTION): chart-03.py $(VALRF)
	$(PYTHON) chart-03.py --data

$(WORKING)/chart-03/max_depth-2004-02.pdf: chart-03.py $(CHART03REDUCTION)
	$(PYTHON) chart-03.py 

# chart-04
CHART04REDUCTION = $(WORKING)/chart-04/data.pickle

$(CHART04REDUCTION): chart-04.py $(VALLIN)
	$(PYTHON) chart-04.py --data

$(WORKING)/chart-04/2004-02.pdf: chart-04.py $(CHART04REDUCTION)
	$(PYTHON) chart-04.py 

# chart-05
CHART05REDUCTION = $(WORKING)/chart-05/data.pickle

$(CHART05REDUCTION): chart-05.py $(VALGBR)
	$(PYTHON) chart-05.py --data

$(WORKING)/chart-05/2004.pdf: chart-05.py $(CHART05REDUCTION)
	$(PYTHON) chart-05.py 

# chart-06 anil
CHART06REDUCTION_ANIL = $(WORKING)/chart-06-anil/data.pickle

$(CHART06REDUCTION_ANIL): chart-06.py $(VALAVM_ANIL)
	$(PYTHON) chart-06.py --valavm anil --data

$(WORKING)/chart-06-anil/a.pdf: chart-06.py $(CHART06REDUCTION_ANIL) $(WORKING)/chart-06-anil/data.pickle
	$(PYTHON) chart-06.py --valavm anil

# chart-06 roy
CHART06REDUCTION_ROY = $(WORKING)/chart-06-roy/data.pickle

$(CHART06REDUCTION_ROY): chart-06.py $(VALAVM_ROY)
	$(PYTHON) chart-06.py --valavm roy --data

$(WORKING)/chart-06-roy/a.pdf: chart-06.py $(CHART06REDUCTION_ANIL) $(WORKING)/chart-06-roy/data.pickle
	$(PYTHON) chart-06.py --valavm roy

# valavm
valavm_dep += valavm.py
valavm_dep += AVM.py
valavm_dep += AVM_gradient_boosting_regressor.py
valavm_dep += AVM_random_forest_regressor.py
valavm_dep += AVM_elastic_net.py
valavm_dep += $(WORKING)/samples-train.csv

# valavm-anil
$(WORKING)/valavm-anil/200612.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200612 anil

$(WORKING)/valavm-anil/200701.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200701 anil

$(WORKING)/valavm-anil/200702.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200702 anil

$(WORKING)/valavm-anil/200703.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200703 anil

$(WORKING)/valavm-anil/200704.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200704 anil

$(WORKING)/valavm-anil/200705.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200705 anil

$(WORKING)/valavm-anil/200706.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200706 anil

$(WORKING)/valavm-anil/200707.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200707 anil

$(WORKING)/valavm-anil/200708.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200708 anil

$(WORKING)/valavm-anil/200709.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200709 anil

$(WORKING)/valavm-anil/200710.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200710 anil

$(WORKING)/valavm-anil/200711.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200711 anil

$(WORKING)/valavm-anil/200712.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200712 anil

# valavm-roy
$(WORKING)/valavm-roy/200612.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200612 roy

$(WORKING)/valavm-roy/200701.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200701 roy

$(WORKING)/valavm-roy/200702.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200702 roy

$(WORKING)/valavm-roy/200703.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200703 roy

$(WORKING)/valavm-roy/200704.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200704 roy

$(WORKING)/valavm-roy/200705.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200705 roy

$(WORKING)/valavm-roy/200706.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200706 roy

$(WORKING)/valavm-roy/200707.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200707 roy

$(WORKING)/valavm-roy/200708.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200708 roy

$(WORKING)/valavm-roy/200709.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200709 roy

$(WORKING)/valavm-roy/200710.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200710 roy

$(WORKING)/valavm-roy/200711.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200711 roy

$(WORKING)/valavm-roy/200712.pickle: $(valavm_dep)
	$(PYTHON) valavm.py 200712 roy


# valgbr
valgbr_dep += valgbr.py 
valgbr_dep += AVM_gradient_boosting_regressor.py
valgbr_dep += $(WORKING)/samples-train.csv
$(WORKING)/valgbr/200402.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200402

$(WORKING)/valgbr/200405.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200405

$(WORKING)/valgbr/200408.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200408

$(WORKING)/valgbr/200411.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200411

$(WORKING)/valgbr/200502.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200502

$(WORKING)/valgbr/200505.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200505

$(WORKING)/valgbr/200508.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200508

$(WORKING)/valgbr/200511.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200511

$(WORKING)/valgbr/200602.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200602

$(WORKING)/valgbr/200605.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200605

$(WORKING)/valgbr/200608.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200608

$(WORKING)/valgbr/200611.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200611

$(WORKING)/valgbr/200702.pickle: $(valgbr_dep) 
	$(PYTHON) valgbr.py 200702

$(WORKING)/valgbr/200705.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200705

$(WORKING)/valgbr/200708.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200708

$(WORKING)/valgbr/200711.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200711

$(WORKING)/valgbr/200802.pickle: $(valgbr_dep) 
	$(PYTHON) valgbr.py 200802

$(WORKING)/valgbr/200805.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200805

$(WORKING)/valgbr/200808.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200808

$(WORKING)/valgbr/200811.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200811

$(WORKING)/valgbr/200902.pickle: $(valgbr_dep)
	$(PYTHON) valgbr.py 200902


# vallin
vallin_dep += vallin.py 
vallin_dep += AVM_elastic_net.py 
vallin_dep += $(WORKING)/samples-train.csv

$(WORKING)/vallin/200402.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200402

$(WORKING)/vallin/200405.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200405

$(WORKING)/vallin/200408.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200408

$(WORKING)/vallin/200411.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200411

$(WORKING)/vallin/200502.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200502

$(WORKING)/vallin/200505.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200505

$(WORKING)/vallin/200508.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200508

$(WORKING)/vallin/200511.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200511

$(WORKING)/vallin/200602.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200602

$(WORKING)/vallin/200605.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200605

$(WORKING)/vallin/200608.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200608

$(WORKING)/vallin/200611.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200611

$(WORKING)/vallin/200702.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200702

$(WORKING)/vallin/200705.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200705

$(WORKING)/vallin/200708.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200708

$(WORKING)/vallin/200711.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200711

$(WORKING)/vallin/200802.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200802

$(WORKING)/vallin/200805.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200805

$(WORKING)/vallin/200808.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200808

$(WORKING)/vallin/200811.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200811

$(WORKING)/vallin/200902.pickle: $(vallin_dep)
	$(PYTHON) vallin.py 200902

# valrf
valrf_dep += valrf.py 
valrf_dep += AVM_random_forest_regressor.py
valrf_dep += $(WORKING)/samples-train.csv

$(WORKING)/valrf/200402.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200402

$(WORKING)/valrf/200405.pickle: $(valrf_dep)
	$(PYTHON) valrf.py 200405

$(WORKING)/valrf/200408.pickle: $(valrf_dep)
	$(PYTHON) valrf.py 200408

$(WORKING)/valrf/200411.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200411

$(WORKING)/valrf/200502.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200502

$(WORKING)/valrf/200505.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200505

$(WORKING)/valrf/200508.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200508

$(WORKING)/valrf/200511.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200511

$(WORKING)/valrf/200602.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200602

$(WORKING)/valrf/200605.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200605

$(WORKING)/valrf/200608.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200608

$(WORKING)/valrf/200611.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200611

$(WORKING)/valrf/200702.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200702

$(WORKING)/valrf/200705.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200705

$(WORKING)/valrf/200708.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200708

$(WORKING)/valrf/200711.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200711

$(WORKING)/valrf/200802.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200802

$(WORKING)/valrf/200805.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200805

$(WORKING)/valrf/200808.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200808

$(WORKING)/valrf/200811.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200811

$(WORKING)/valrf/200902.pickle: $(valrf_dep) 
	$(PYTHON) valrf.py 200902

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
