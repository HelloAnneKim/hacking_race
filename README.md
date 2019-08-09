# hacking_race
problems with "ancestry reports"

## BioHacking Village
* https://www.villageb.io/
* https://www.eventbrite.ca/e/bhv-workshop-hacking-race-problems-with-ancestry-reports-tickets-65745310995
* 11:45am-12:45pm PST Friday August 8, 2019

* Have you ever wondered how "race" and ancestry are "calculated" by 23andMe or Ancestry.com? Have you ever questioned why your results seemed so "wrong"? Well, in this workshop we'll be building our own admixture (ancestry) reports from publicly available data as well as showing how we can "hack race" to make racial results different.
* The respository is based on modified code from https://github.com/hagax8/ancestry_viz 

*Make sure you have git, python, and pip installed*

### Step 0 (Optional): Make a virtual environment 
This part is optional but encouraged because we're about to download files and stuff that might mess up existing dependencies you have in your dev environment.  We'll be following the instructions from: https://docs.python-guide.org/dev/virtualenvs/

`$ cd ~`

`$ sudo apt install virtualenv`

`$ cd <directory_path_you_work_in>` ie. `$ cd Downloads` or something

`$ virtualenv -p /usr/bin/python2.7 venv` to make a virtual environment named `venv`

`$ source venv/bin/activate` to activate the virtual environment 

### Step 1: CLONE THIS REPO 

`$ cd venv`

`$ git clone https://github.com/herroannekim/hacking_race.git` into your virtual environment 

### Step 2: Download the Data

`$ ./download_viz.sh`

### Step 3: Install dependencies
Either make a virtual environment or install the dependencies to your system: 

`$ pip install -r requirements.txt`

### Step 4: Run ancestry prediction and clustering
In this example, we're running a prediction on sample HG01108, a Puerto Rican female

http://www.internationalgenome.org/data-portal/sample/HG01108

`$ python2  hacker.py  --model GTM  --out classify_HG01108 --config ./standard_config.json  --classify-id HG01108`

### Step 5: Hack 3:D
For the ancestry hack, we will now shift her ancestry towards Kenyan using

`$ python2 hacker.py --model GTM --out kenya_dig_it --classify-id HG01108 --config ./standard_config.json --manipulate-towards "Luhya_in_Webuye,_Kenya"`

### Step 6: Check your work
* Now you should be able to check the output and cluster graphs :)

### Step 7 (optional): Clean up
If you used a virtual environment, you can clean up with
`$ deactivate`





