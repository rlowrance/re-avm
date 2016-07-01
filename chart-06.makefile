# create chart-06 anil and roy outputs
PYTHON = ~/anaconda/bin/python

WORKING = ../data/working

.PHONY : all

all: anil-data anil-charts roy-data roy-charts

anil-data: chart-06.py 
	$(PYTHON) chart-06.py --valavm anil --data

anil-charts: chart-06.py  $(WORKING)/chart-06-anil/data.pickle
	$(PYTHON) chart-06.py --valavm anil

roy-data: chart-06.py
	$(PYTHON) chart-06.py --valavm roy --data

roy-charts: chart-06.py  $(WORKING)/chart-06-roy/data.pickle
	$(PYTHON) chart-06.py --valavm roy
