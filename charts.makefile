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
