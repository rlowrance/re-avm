# --debug=basic
# disable built-in rules
.SUFFIXES:



PYTHON = ~/anaconda/bin/python

INPUT = ../data/input
WORKING = ../data/working
CVCELL = ../data/working/cv-cell
CVCELLRESCALED = ../data/working/cv-cell-rescaled

INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_07/CAC06037F1.zip
INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_07/CAC06037F2.zip
INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_07/CAC06037F3.zip
INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_07/CAC06037F4.zip
INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_09/CAC06037F5.zip
INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_09/CAC06037F6.zip
INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_09/CAC06037F7.zip
INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_09/CAC06037F8.zip

INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F1.zip
INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F2.zip
INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F3.zip
INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F4.zip
INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F5.zip
INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F6.zip
INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F7.zip
INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F8.zip

INPUT_CENSUS += $(INPUT)/neighborhood-data/census.csv

#ALL += $(WORKING)/census.RData
#ALL += $(WORKING)/deeds-al-g.RData
#ALL += $(WORKING)/parcels-coded.RData
#ALL += $(WORKING)/parcels-derived-features.RData
#ALL += $(WORKING)/parcels-sfr.RData
#ALL += $(WORKING)/transactions.RData
#ALL += $(WORKING)/transactions-subset2.csv
ALL += $(WORKING)/transactions-subset2.pickle
ALL += $(WORKING)/transactions-subset2-test.pickle
ALL += $(WORKING)/transactions-subset2-train.pickle
ALL += $(WORKING)/transactions-subset2-rescaled.pickle
ALL += $(WORKING)/transactions-subset2-rescaled-test.pickle
ALL += $(WORKING)/transactions3-al-g-sfr.csv
ALL += $(WORKING)/transactions3-subset-test.csv
ALL += $(WORKING)/transactions3-subset-train.csv
#ALL += $(WORKING)/transactions-subset2-rescaled-train.pickle
ALL += $(WORKING)/chart-01.pdf
ALL += $(WORKING)/chart-02-ols-2003on-ct-t-mean-mean.txt
ALL += $(WORKING)/chart-02-ols-2003on-ct-t-mean-wi10.txt
ALL += $(WORKING)/chart-02-ols-2003on-ct-t-median-median.txt
ALL += $(WORKING)/chart-02-ransac-2003on-ct-t-mean-mean.txt
ALL += $(WORKING)/chart-02-ransac-2003on-ct-t-mean-wi10.txt
ALL += $(WORKING)/chart-02-ransac-2003on-ct-t-median-median.txt
ALL += $(WORKING)/chart-02-ols-2008-act-ct-mean-mean.txt
ALL += $(WORKING)/chart-02-ols-2008-act-ct-median-median.txt
ALL += $(WORKING)/chart-02-ransac-2008-act-ct-median-median.txt
ALL += $(WORKING)/chart-03.txt
ALL += $(WORKING)/chart-04.natural.nz-count-all-periods.txt


ALL += $(WORKING)/ege_date-2009-02-16.pickle
ALL += $(WORKING)/ege_to_dataframe-2009-02-16.pickle
ALL += $(WORKING)/ege_summary_by_scope-2009-02-16.pickle
ALL += $(WORKING)/chart-05.txt

ALL += ege_week.makefile


#ALL += $(WORKING)/chart-04.rescaled.nz-count-all-periods.txt
#ALL += $(WORKING)/record-counts.tex
#ALL += $(WORKING)/python-dependencies.makefile

all: $(ALL)
include $(WORKING)/python-dependencies.makefile

$(WORKING)/python-dependencies.makefile: python-dependencies.py

$(WORKING)/transactions3-al-g-sfr.csv: transactions3.py
	$(PYTHON) transactions3.py

$(WORKING)/transactions3-subset-test%csv $(WORKING)/transactions3-subset-train%csv: \
	transactions3-subset.py
	$(PYTHON) transactions3-subset.py

# ege_week files; STEM is .

ege_week.makefile: ege_week_makefile.py
	$(PYTHON) ege_week_makefile.py

$(WORKING)/ege_week-2009-02-15-df-test%pickle $(WORKING)/ege_week-2009-02-15-dict-test%pickle: \
	ege_week.py $(WORKING)/transactions-subset2.pickle
	$(PYTHON) ege_week.py 2009-02-15 --testing

$(WORKING)/ege_week-2009-02-15-df%pickle $(WORKING)/ege_week-2009-02-15-dict%pickle: \
	ege_week.py $(WORKING)/transactions-subset2.pickle
	$(PYTHON) ege_week.py 2009-02-15 --global

# ege_date files; STEM is the sale_date

$(WORKING)/ege_summary_by_scope-%.pickle: ege_summary_by_scope.py $(WORKING)/ege_to_dataframe-%.pickle
	$(PYTHON) ege_summary_by_scope.py $*

$(WORKING)/ege_to_dataframe-%.pickle: ege_to_dataframe.py $(WORKING)/ege_date-%.pickle
	$(PYTHON) ege_to_dataframe.py $*

$(WORKING)/ege_date-%.pickle: ege_date.py
	$(PYTHON) ege_date.py $*

$(WORKING)/chart-05.txt: chart-05.py $(WORKING)/ege_summary_by_scope-2009-02-16.pickle
	$(PYTHON) chart-05.txt 2009-02-16

# Creation of cvcell
$(CVCELL)/%.cvcell:
	$(PYTHON) cv-cell.py $*

# rules for CHARTS
include chart-01.makefile
include chart-02-ols-2003on-ct-t-mean-mean.makefile
include chart-02-ols-2003on-ct-t-mean-wi10.makefile
include chart-02-ols-2003on-ct-t-median-median.makefile
include chart-02-ransac-2003on-ct-t-mean-mean.makefile
include chart-02-ransac-2003on-ct-t-mean-wi10.makefile
include chart-02-ransac-2003on-ct-t-median-median.makefile
include chart-02-ols-2008-act-ct-mean-mean.makefile
include chart-02-ols-2008-act-ct-median-median.makefile
include chart-02-ransac-2008-act-ct-median-median.makefile
include chart-03.makefile

# rules for other *.makefile files
chart-02-ols-2003on-ct-t-mean-mean.makefile: \
	chart-02-ols-2003on-ct-t-mean-mean.py
	python chart-02-ols-2003on-ct-t-mean-mean.py makefile

chart-02-ols-2003on-ct-t-mean-wi10.makefile: \
	chart-02-ols-2003on-ct-t-mean-wi10.py
	python chart-02-ols-2003on-ct-t-mean-wi10.py makefile

chart-02-ols-2003on-ct-t-median-median.makefile: \
	chart-02-ols-2003on-ct-t-median-median.py
	python chart-02-ols-2003on-ct-t-median-median.py makefile

chart-02-ransac-2003on-ct-t-mean-mean.makefile: \
	chart-02-ransac-2003on-ct-t-mean-mean.py
	python chart-02-ransac-2003on-ct-t-mean-mean.py makefile

chart-02-ransac-2003on-ct-t-mean-wi10.makefile: \
	chart-02-ransac-2003on-ct-t-mean-wi10.py
	python chart-02-ransac-2003on-ct-t-mean-wi10.py makefile

chart-02-ransac-2003on-ct-t-median-median.makefile: \
	chart-02-ransac-2003on-ct-t-median-median.py
	python chart-02-ransac-2003on-ct-t-median-median.py makefile

chart-02-ols-2008-act-ct-mean-mean.makefile: \
  chart-02-ols-2008-act-ct-mean-mean.py 
	python chart-02-ols-2008-act-ct-mean-mean.py makefile

chart-02-ols-2008-act-ct-median-median.makefile: \
  chart-02-ols-2008-act-ct-median-median.py 
	python chart-02-ols-2008-act-ct-median-median.py makefile

chart-02-ransac-2008-act-ct-median-median.makefile: \
  chart-02-ransac-2008-act-ct-median-median.py 
	python chart-02-ransac-2008-act-ct-median-median.py makefile

#chart-02-huber100-median-of-root-median-squared-errors.makefile: \
#  chart-02-huber100-median-of-root-median-squared-errors.py 
#	python chart-02-huber100-median-of-root-median-squared-errors.py makefile
#
#chart-02-theilsen-median-of-root-median-squared-errors.makefile: \
#  chart-02-theilsen-median-of-root-median-squared-errors.py 
#	python chart-02-theilsen-median-of-root-median-squared-errors.py makefile

# chart 04

c4cellspec = lassocv-logprice-ct-2003on-30
c4cvcellnatural = $(CVCELL)/$(c4cellspec).cvcell
c4unitsnatural = natural
c4examplenatural = $(WORKING)/chart-04.natural.nz-count-all-periods.txt

# chart 05

# create cell in natural units 
# NOTE: don't depend on cv-cell.py, as it changes all the time
# and cell creation take a long time
$(c4cvcellnatural): $(transactionsnatural)
	python cv-cell.py\
		$(c4cellspec) \
		--in $(transactionsnatural) \
		--out $(c4cvcellnatural)

# the target is an example
# running the recipe creates multiple targets
$(c4examplenatural): chart-04.py $(c4cvcellnatural)
	python chart-04.py \
		--in $(c4cvcellnatural) \
		--cache \
		--units natural

c4cvcellrescaled = $(CVCELLRESCALED)/$(c4cellspec).cvcell
c4unitsrescaled = rescaled
c4examplerescaled = $(WORKING)/chart-04.$(c4unitsrescaled).nz-count-all-periods.txt
transactionsrescaled = $(WORKING)/transactions-subset2-rescaled.pickle
#$(info c4cvcellrescaled       $(c4cvcellrescaled))
#$(info c4unitsrescaled        $(c4unitsrescaled))
#$(info c4examplerescaled      $(c4example))
#$(info c4transactionsrescaled $(c4transactionsrescaled))

$(c4cvcellrescaled): $(transactionsrescaled)
	python cv-cell.py \
		$(c4cellspec) \
		--in $(transactionsrescaled) \
		--out $(c4cvcellrescaled) \
		--age no

$(c4examplerescaled): chart-04.py $(c4cvcellrescaled)
	python chart-04.py \
		--in $(c4cvcellrescaled) \
		--cache \
		--units $(c4unitsrescaled)



# recipe to delete the cache and chart files
# TODO: delete all the chart-04 files, after replicated prior result
.PHONY: clean-04
clean-04:
	rm $(WORKING)/chart-04.*


# GENERATED TEX FILES

$(WORKING)/record-counts.tex: \
$(WORKING)/parcels-sfr-counts.csv \
$(WORKING)/deeds-al-g-counts.csv \
$(WORKING)/transactions-counts.csv \
$(WORKING)/transactions-subset2-counts.csv \
record-counts.py
	python record-counts.py


# DATA
$(WORKING)/census.RData: $(INPUT_CENSUS) census.R
	Rscript census.R

$(WORKING)/deeds-al-g%RData \
$(WORKING)/deeds-al-g-counts%csv \
: $(INPUT_DEEDS) deeds-al-g.R
	Rscript deeds-al-g.R

$(WORKING)/parcels-coded.RData: $(INPUT_TAXROLLS) parcels-coded.R
	RScript parcels-coded.R

$(WORKING)/parcels-derived-features.RData: $(INPUT_TAXROLLS) parcels-derived-features.R
	RScript parcels-derived-features.R

$(WORKING)/parcels-sfr%RData \
$(WORKING)/parcels-sfr-counts%csv \
: $(INPUT_TAXROLLS) parcels-sfr.R
	Rscript parcels-sfr.R

$(WORKING)/transactions%RData $(WORKING)/transactions%csv:\
	$(WORKING)/census.RData \
	$(WORKING)/deed-al-g.RData \
	$(INPUT)/geocoding.tsv \
	$(WORKING)/parcels-derived-features.RData \
	$(WORKING)/parcels-sfr.RData \
	transactions.R
	Rscript transactions.R

# transactions subsets: WORKING/transactions-subset2-VARIANTS.KINDS

$(WORKING)/transactions-subset2.pickle $(WORKING)/transactions-subset2-counts.csv: \
	$(WORKING)/transactions.csv transactions-subset2.py
	python transactions-subset2.py

$(WORKING)/transactions-subset2.csv: \
	unpickle-transactions-subset2.py $(WORKING)/transactions-subset2.pickle
	python unpickle-transactions-subset2.py 

$(WORKING)/transactions-subset2-test%pickle $(WORKING)/transactions-subset2-train%pickle:\
	$(WORKING)/transactions-subset2.pickle split.py
	python split.py \
		--test     0.10 \
		--in       $(WORKING)/transactions-subset2.pickle \
		--outtest  $(WORKING)/transactions-subset2-test.pickle \
		--outtrain $(WORKING)/transactions-subset2-train.pickle

# transactions subsets: WORKING/transactions-subset2-rescaled-VARIANTS.pickle

ts2prefix = $(WORKING)/transactions-subset2
$(ts2prefix)-rescaled.pickle: rescale.py $(ts2prefix).pickle
	python rescale.py \
		--in $(ts2prefix).pickle \
		--out $(ts2prefix)-rescaled.pickle

ts2rescaledprefix = $(WORKING)/transactions-subset2-rescaled
#$info ts2rescaledprefix $(ts2rescaledprefix))
$(ts2rescaledprefix)-test%pickle $(ts2rescaledprefix)-train%pickle: \
	rescale.py $(ts2rescaledprefix).pickle
	python split.py \
		--test     0.10 \
		--in       $(ts2rescaledprefix).pickle \
		--outtest  $(ts2rescaledprefix)-test.pickle \
		--outtrain $(ts2rescaledprefix)-train.pickle


# source file dependencies R language
census.R: \
	Directory.R InitializeR.R
deeds-al-g.R: \
	Directory.R InitializeR.R DEEDC.R Printf.R PRICATCODE.R
parcels-coded.R: \
	Directory.R InitializeR.R LUSEI.R PROPN.R ReadRawParcels.R
parcels-derived-features.R: \
	Directory.R InitializeR.R Methods.R Clock.R LUSEI.R Printf.R PROPN.R ReadParcelsCoded.R ZipN.R
parcels-sfr.R: \
	Directory.R InitializeR.R LUSEI.R Printf.R ReadRawParcels.R
transactions.R: \
	Directory.R InitializeR.R BestApns.R ReadCensus.R ReadDeedsAlG.R ReadParcelsSfr.R ZipN.R
