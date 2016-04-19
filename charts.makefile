# define how to make charts used in paper1
PYTHON = ~/anaconda/bin/python

WORKING = ../data/working
# name one chart from each set of used chart
# Note: many charts supported preliminary analysis not in the final paper
CHARTS += $(WORKING)/chart01/median-price.pdf
CHARTS += $(WORKING)/chart06/a.pdf

.PHONY : all-charts
all-charts: $(CHARTS)


# chart01
$(WORKING)/chart01/data.pickle: chart01.py $(WORKING)/samples-train-validate.csv
	$(PYTHON) chart01.py --data

$(WORKING)/chart01/median-price.pdf: chart01.py $(WORKING)/chart01/data.pickle
	$(PYTHON) chart01.py
	
# chart06 
DEPS =
DEPS += $(WORKING)/valavm/200612.pickle
DEPS += $(WORKING)/valavm/200701.pickle
DEPS += $(WORKING)/valavm/200702.pickle
DEPS += $(WORKING)/valavm/200703.pickle
DEPS += $(WORKING)/valavm/200704.pickle
DEPS += $(WORKING)/valavm/200705.pickle
DEPS += $(WORKING)/valavm/200706.pickle
DEPS += $(WORKING)/valavm/200707.pickle
DEPS += $(WORKING)/valavm/200708.pickle
DEPS += $(WORKING)/valavm/200709.pickle
DEPS += $(WORKING)/valavm/200710.pickle
DEPS += $(WORKING)/valavm/200711.pickle
DEPS += $(WORKING)/valavm/200712.pickle
DEPS += $(WORKING)/valavm/200801.pickle
DEPS += $(WORKING)/valavm/200802.pickle
DEPS += $(WORKING)/valavm/200803.pickle
DEPS += $(WORKING)/valavm/200804.pickle
DEPS += $(WORKING)/valavm/200805.pickle

$(WORKING)/chart06/data.pickle: chart06.py $(WORKING)/chart01/data.pickle $(DEPS)
	$(PYTHON) chart06.py  --data

$(WORKING)/chart06/a.pdf: chart06.py $(WORKING)/chart06/data.pickle
	$(PYTHON) chart06.py 

# BELOW ME ARE HISTORIC CHARTS
# chart-06 anil (HISTORIC)
CHART06REDUCTION_ANIL = $(WORKING)/chart-06-anil/data.pickle

$(CHART06REDUCTION_ANIL): chart-06.py $(VALAVM_ANIL)
	$(PYTHON) chart-06.py --valavm anil --data

$(WORKING)/chart-06-anil/a.pdf: chart-06.py $(CHART06REDUCTION_ANIL) $(WORKING)/chart-06-anil/data.pickle
	$(PYTHON) chart-06.py --valavm anil

# chart-06 roy (HISTORIC)
CHART06REDUCTION_ROY = $(WORKING)/chart-06-roy/data.pickle

$(CHART06REDUCTION_ROY): chart-06.py $(VALAVM_ROY)
	$(PYTHON) chart-06.py --valavm roy --data

$(WORKING)/chart-06-roy/a.pdf: chart-06.py $(CHART06REDUCTION_ANIL) $(WORKING)/chart-06-roy/data.pickle
	$(PYTHON) chart-06.py --valavm roy
	

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

