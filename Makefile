install:
	python3 -m pip install -r requirements.txt
	cd scripts
	chmod u+x getdata.sh
	./getdata.sh
hide:
	chmod u+x scripts/editconstant.sh
	scripts/editconstant.sh hide
show:
	chmod u+x scripts/editconstant.sh
	scripts/editconstant.sh show
