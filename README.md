# hacking_race
problems with "ancestry reports"

## BioHacking Village
* https://www.villageb.io/
* https://www.eventbrite.ca/e/bhv-workshop-hacking-race-problems-with-ancestry-reports-tickets-65745310995
* 11:45am-12:45pm PST Friday August 8, 2019

* Have you ever wondered how "race" and ancestry are "calculated" by 23andMe or Ancestry.com? Have you ever questioned why your results seemed so "wrong"? Well, in this workshop we'll be building our own admixture (ancestry) reports from publicly available data as well as showing how we can "hack race" to make racial results different.
* The respository is based on modified code from https://github.com/hagax8/ancestry_viz 

### CLONE THIS REPO 
`$ git clone https://github.com/herroannekim/hacking_race.git`

### Step 0 (Optional): Make a virtual environment 
This part is optional but encouraged because we're about to download files and stuff that might mess up existing dependencies you have in your dev environment.  We'll be following the instructions from: https://docs.python-guide.org/dev/virtualenvs/

`$ cd hacking_race`

`$ pip install virtualenv`

`$ virtualenv venv`

`$ virtualenv -p /usr/bin/python2.7 venv`

`$ source venv/bin/activate`

### Step 1: Download the Data
`$ chmod +x download_viz.sh`

`$ ./download_viz.sh`

### Step 2: Install dependencies
Either make a virtual environment or install the dependencies to your system: 
`$ pip install -r requirements.txt`

### Step 3: Run ancestry prediction and clustering
In this example, we're running a prediction on sample HG01108, a Puerto Rican female

http://www.internationalgenome.org/data-portal/sample/HG01108

`$ python2  hacker.py  --model GTM  --out classify_HG01108 --config ./standard_config.json  --classify-id HG01108`

### Step 4: Hack 3:D
For the ancestry hack, we will now shift her ancestry towards Kenyan using

`$ python2 hacker.py --model GTM --out kenya_dig_it --classify-id HG01108 --config ./standard_config.json --manipulate-towards "Luhya_in_Webuye,_Kenya"`

### Step 5: Check your work
* Now you should be able to check the output and cluster graphs :)

### Step 6 (optional): Clean up
If you used a virtual environment, you can clean up with
`$ deactivate`





