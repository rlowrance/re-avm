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
# HP210z has 2, 4T, 16 GB 
VALAVM_S_ALL_HP     += $(WORKING)/valavm/s-all/s-all-200512.pickle
VALAVM_S_ALL_HP     += $(WORKING)/valavm/s-all/s-all-200601.pickle
VALAVM_S_ALL_HP     += $(WORKING)/valavm/s-all/s-all-200602.pickle
VALAVM_S_ALL_HP     += $(WORKING)/valavm/s-all/s-all-200603.pickle
VALAVM_S_ALL_ROY    += $(WORKING)/valavm/s-all/s-all-200604.pickle
VALAVM_S_ALL_ROY    += $(WORKING)/valavm/s-all/s-all-200605.pickle
VALAVM_S_ALL_ROY    += $(WORKING)/valavm/s-all/s-all-200606.pickle
VALAVM_S_ALL_ROY    += $(WORKING)/valavm/s-all/s-all-200607.pickle
VALAVM_S_ALL_ROY    += $(WORKING)/valavm/s-all/s-all-200608.pickle
VALAVM_S_ALL_ROY    += $(WORKING)/valavm/s-all/s-all-200609.pickle
VALAVM_S_ALL_ROY    += $(WORKING)/valavm/s-all/s-all-200610.pickle
VALAVM_S_ALL_ROY    += $(WORKING)/valavm/s-all/s-all-200611.pickle
VALAVM_S_ALL_ROY    += $(WORKING)/valavm/s-all/s-all-200612.pickle
VALAVM_S_ALL_ROY    += $(WORKING)/valavm/s-all/s-all-200701.pickle
VALAVM_S_ALL_ROY    += $(WORKING)/valavm/s-all/s-all-200702.pickle
VALAVM_S_ALL_ROY    += $(WORKING)/valavm/s-all/s-all-200703.pickle
VALAVM_S_ALL_HP     += $(WORKING)/valavm/s-all/s-all-200704.pickle
VALAVM_S_ALL_HP     += $(WORKING)/valavm/s-all/s-all-200705.pickle
VALAVM_S_ALL_HP     += $(WORKING)/valavm/s-all/s-all-200706.pickle
VALAVM_S_ALL_HP     += $(WORKING)/valavm/s-all/s-all-200707.pickle
VALAVM_S_ALL_HP     += $(WORKING)/valavm/s-all/s-all-200708.pickle
VALAVM_S_ALL_HP     += $(WORKING)/valavm/s-all/s-all-200709.pickle
VALAVM_S_ALL_HP     += $(WORKING)/valavm/s-all/s-all-200710.pickle
VALAVM_S_ALL_HP     += $(WORKING)/valavm/s-all/s-all-200711.pickle
VALAVM_S_ALL_JUDITH += $(WORKING)/valavm/s-all/s-all-200712.pickle
VALAVM_S_ALL_JUDITH += $(WORKING)/valavm/s-all/s-all-200801.pickle
VALAVM_S_ALL_JUDITH += $(WORKING)/valavm/s-all/s-all-200802.pickle
VALAVM_S_ALL_JUDITH += $(WORKING)/valavm/s-all/s-all-200803.pickle
VALAVM_S_ALL_JUDITH += $(WORKING)/valavm/s-all/s-all-200804.pickle
VALAVM_S_ALL_JUDITH += $(WORKING)/valavm/s-all/s-all-200805.pickle
VALAVM_S_ALL_JUDITH += $(WORKING)/valavm/s-all/s-all-200806.pickle
VALAVM_S_ALL_JUDITH += $(WORKING)/valavm/s-all/s-all-200807.pickle
VALAVM_S_ALL_X      += $(WORKING)/valavm/s-all/s-all-200808.pickle
VALAVM_S_ALL_X      += $(WORKING)/valavm/s-all/s-all-200809.pickle
VALAVM_S_ALL_X      += $(WORKING)/valavm/s-all/s-all-200810.pickle
VALAVM_S_ALL_X      += $(WORKING)/valavm/s-all/s-all-200811.pickle
VALAVM_S_ALL_X      += $(WORKING)/valavm/s-all/s-all-200812.pickle
VALAVM_S_ALL_X      += $(WORKING)/valavm/s-all/s-all-200901.pickle
VALAVM_S_ALL_X      += $(WORKING)/valavm/s-all/s-all-200902.pickle
VALAVM_S_ALL += $(VALAVM_S_ALL_DELL)
VALAVM_S_ALL += $(VALAVM_S_ALL_ROY)
VALAVM_S_ALL += $(VALAVM_S_ALL_HP)
VALAVM_S_ALL += $(VALAVM_S_ALL_JUDITH)
VALAVM_S_ALL += $(VALAVM_S_ALL_X)
ALL += $(VALAVM_S_ALL)

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
VALAVM_SW_ALL_JUDITH += $(WORKING)/valavm/sw-all/sw-all-200802.pickle
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

VALAVM_SWP_ALL_HP     += $(WORKING)/valavm/swp-all/swp-all-200512.pickle
VALAVM_SWP_ALL_HP     += $(WORKING)/valavm/swp-all/swp-all-200601.pickle
VALAVM_SWP_ALL_HP     += $(WORKING)/valavm/swp-all/swp-all-200602.pickle
VALAVM_SWP_ALL_HP     += $(WORKING)/valavm/swp-all/swp-all-200603.pickle

VALAVM_SWP_ALL_ROY    += $(WORKING)/valavm/swp-all/swp-all-200604.pickle
VALAVM_SWP_ALL_ROY    += $(WORKING)/valavm/swp-all/swp-all-200605.pickle
VALAVM_SWP_ALL_ROY    += $(WORKING)/valavm/swp-all/swp-all-200606.pickle
VALAVM_SWP_ALL_ROY    += $(WORKING)/valavm/swp-all/swp-all-200607.pickle
VALAVM_SWP_ALL_ROY    += $(WORKING)/valavm/swp-all/swp-all-200608.pickle
VALAVM_SWP_ALL_ROY    += $(WORKING)/valavm/swp-all/swp-all-200609.pickle
VALAVM_SWP_ALL_ROY    += $(WORKING)/valavm/swp-all/swp-all-200610.pickle
VALAVM_SWP_ALL_ROY    += $(WORKING)/valavm/swp-all/swp-all-200611.pickle
VALAVM_SWP_ALL_ROY    += $(WORKING)/valavm/swp-all/swp-all-200612.pickle
VALAVM_SWP_ALL_ROY    += $(WORKING)/valavm/swp-all/swp-all-200701.pickle
VALAVM_SWP_ALL_ROY    += $(WORKING)/valavm/swp-all/swp-all-200702.pickle
VALAVM_SWP_ALL_ROY    += $(WORKING)/valavm/swp-all/swp-all-200703.pickle

VALAVM_SWP_ALL_JUDITH += $(WORKING)/valavm/swp-all/swp-all-200704.pickle
VALAVM_SWP_ALL_JUDITH += $(WORKING)/valavm/swp-all/swp-all-200705.pickle
VALAVM_SWP_ALL_JUDITH += $(WORKING)/valavm/swp-all/swp-all-200706.pickle
VALAVM_SWP_ALL_JUDITH += $(WORKING)/valavm/swp-all/swp-all-200707.pickle
VALAVM_SWP_ALL_JUDITH += $(WORKING)/valavm/swp-all/swp-all-200708.pickle
VALAVM_SWP_ALL_JUDITH += $(WORKING)/valavm/swp-all/swp-all-200709.pickle
VALAVM_SWP_ALL_JUDITH += $(WORKING)/valavm/swp-all/swp-all-200710.pickle

VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200711.pickle
VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200712.pickle
VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200801.pickle
VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200802.pickle
VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200803.pickle
VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200804.pickle
VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200805.pickle
VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200806.pickle
VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200807.pickle
VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200808.pickle
VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200809.pickle
VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200810.pickle
VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200811.pickle
VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200812.pickle
VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200901.pickle
VALAVM_SWP_ALL_DELL   += $(WORKING)/valavm/swp-all/swp-all-200902.pickle

VALAVM_SWP_ALL += $(VALAVM_SWP_ALL_DELL)
VALAVM_SWP_ALL += $(VALAVM_SWP_ALL_ROY)
VALAVM_SWP_ALL += $(VALAVM_SWP_ALL_HP)
VALAVM_SWP_ALL += $(VALAVM_SWP_ALL_JUDITH)
VALAVM_SWP_ALL += $(VALAVM_SWP_ALL_X)

ALL += $(VALAVM_SWP_ALL)

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
CHART01 += $(WORKING)/chart01/median-price.pdf
CHART06 += $(WORKING)/chart06/s-all/a.pdf
CHART06 += $(WORKING)/chart06/sw-all/a.pdf
CHART06 += $(WORKING)/chart06/swpn-all/a.pdf

CHART07 += $(WORKING)/chart07/s-all/b.txt
CHART07 += $(WORKING)/chart07/sw-all/b.txt
#CHART07 += $(WORKING)/chart07/swp-all/b.txt
CHART07 += $(WORKING)/chart07/swpn-all/b.txt

ALLCHARTS = $(CHART01) $(CHART06) $(CHART07)

ALL += $(ALLCHARTS)

# pick a representative of all the rank_models
#ALL += $(WORKING)/rank_models/200512.pickle

.PHONY : all
all: $(ALL)

.PHONY : dell-s-all
dell-s-all: $(VALAVM_S_ALL)

.PHONY : dell-swpn-all
dell-swpn-all: $(VALAVM_SWPN_ALL_DELL)

.PHONY : roy-s-all
roy-s-all: $(VALAVM_S_ALL)

.PHONY : roy-sw-all
roy-sw-all: $(VALAVM_SW_ALL_ROY)

.PHONY : valavm-swp-dell valavm-swp-judith valavm-swp-hp valavm-swpn-roy
valavm-swp-dell: $(VALAVM_SWP_ALL_DELL)
valavm-swp-judith: $(VALAVM_SWP_ALL_JUDITH)
valavm-swp-hp: $(VALAVM_SWP_ALL_HP)
valavm-swp-roy: $(VALAVM_SWP_ALL_ROY)

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

.PHONY: chart01
chart01: $(CHART01)

.PHONY: chart06
chart06: $(CHART06)

.PHONY: chart07
chart07: $(CHART07)

# census-features-derived.csv
$(WORKING)/census-features-derived.csv: census-features.py layout_census.py
	$(PYTHON) census-features.py


# builds for charts actually used
# NOTE: some charts were created and not used ihe final report

# chart01
$(WORKING)/chart01/data.pickle: chart01.py $(WORKING)/samples-train.csv
	$(PYTHON) chart01.py --data

$(WORKING)/chart01/median-price.pdf: chart01.py $(WORKING)/chart01/data.pickle
	$(PYTHON) chart01.py
	
# chart06 
$(WORKING)/chart06/s-all/data.pickle: chart06.py $(WORKING)/chart01/data.pickle $(VALAVM)
	$(PYTHON) chart06.py s-all --data

$(WORKING)/chart06/s-all/a.pdf: chart06.py $(WORKING)/chart06/s-all/data.pickle
	$(PYTHON) chart06.py s-all

$(WORKING)/chart06/sw-all/data.pickle: chart06.py $(WORKING)/chart01/data.pickle $(VALAVM)
	$(PYTHON) chart06.py sw-all --data

$(WORKING)/chart06/sw-all/a.pdf: chart06.py $(WORKING)/chart06/sw-all/data.pickle
	$(PYTHON) chart06.py sw-all

$(WORKING)/chart06/swpn-all/data.pickle: chart06.py $(WORKING)/chart01/data.pickle $(VALAVM)
	$(PYTHON) chart06.py swpn-all --data

$(WORKING)/chart06/swpn-all/a.pdf: chart06.py $(WORKING)/chart06/swpn-all/data.pickle
	$(PYTHON) chart06.py swpn-all

# chart07
$(WORKING)/chart07/s-all/data.pickle: chart07.py $(VALAVM_FITTED)
	$(PYTHON) chart07.py s-all --data

$(WORKING)/chart07/s-all/b.txt: chart07.py $(WORKING)/chart07/s-all/data.pickle
	$(PYTHON) chart07.py s-all

$(WORKING)/chart07/sw-all/data.pickle: chart07.py $(VALAVM_FITTED)
	$(PYTHON) chart07.py sw-all --data

$(WORKING)/chart07/sw-all/b.txt: chart07.py $(WORKING)/chart07/sw-all/data.pickle
	$(PYTHON) chart07.py sw-all

$(WORKING)/chart07/swpn-all/data.pickle: chart07.py $(VALAVM_FITTED)
	$(PYTHON) chart07.py swpn-all --data

$(WORKING)/chart07/swpn-all/b.txt: chart07.py $(WORKING)/chart07/swpn-all/data.pickle
	$(PYTHON) chart07.py swpn-all

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


