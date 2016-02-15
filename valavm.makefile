PYTHON = ~/anaconda/bin/python

# setup makefile so that jobs can run in parallel using -j N invocation option
.PHONY : all
all: a b c d e f g h i j k l

a:
	$(PYTHON) valavm.py 200612 anil

b:
	$(PYTHON) valavm.py 200701 anil

c:
	$(PYTHON) valavm.py 200702 anil

d:
	$(PYTHON) valavm.py 200703 anil

e:
	$(PYTHON) valavm.py 200704 anil

f:
	$(PYTHON) valavm.py 200705 anil

g:
	$(PYTHON) valavm.py 200706 anil

h:
	$(PYTHON) valavm.py 200707 anil

i:
	$(PYTHON) valavm.py 200708 anil

j:
	$(PYTHON) valavm.py 200709 anil

k:
	$(PYTHON) valavm.py 200710 anil

l:
	$(PYTHON) valavm.py 200711 anil
