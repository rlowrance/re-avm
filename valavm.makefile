PYTHON = ~/anaconda/bin/python

# setup makefile so that jobs can run in parallel using -j N invocation option
.PHONY : all
all: a b c d e f g h i j k l

a:
	$(PYTHON) valavm.py 200612 --out ../data/working/valavm-anil/200612.pickle

b:
	$(PYTHON) valavm.py 200701 --out ../data/working/valavm-anil/200701.pickle

c:
	$(PYTHON) valavm.py 200702 --out ../data/working/valavm-anil/200702.pickle

d:
	$(PYTHON) valavm.py 200703 --out ../data/working/valavm-anil/200703.pickle

e:
	$(PYTHON) valavm.py 200704 --out ../data/working/valavm-anil/200704.pickle

f:
	$(PYTHON) valavm.py 200705 --out ../data/working/valavm-anil/200705.pickle

g:
	$(PYTHON) valavm.py 200706 --out ../data/working/valavm-anil/200706.pickle

h:
	$(PYTHON) valavm.py 200707 --out ../data/working/valavm-anil/200707.pickle

i:
	$(PYTHON) valavm.py 200708 --out ../data/working/valavm-anil/200708.pickle

j:
	$(PYTHON) valavm.py 200709 --out ../data/working/valavm-anil/200709.pickle

k:
	$(PYTHON) valavm.py 200710 --out ../data/working/valavm-anil/200710.pickle

l:
	$(PYTHON) valavm.py 200711 --out ../data/working/valavm-anil/200711.pickle
