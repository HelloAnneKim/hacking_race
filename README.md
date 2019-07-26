# hacking_race
problems with "ancestry reports"

## BioHacking Village
* https://www.villageb.io/
* 11am-12pm PST Friday August 8, 2019

* Have you ever wondered how "race" and ancestry are "calculated" by 23andMe or Ancestry.com? Have you ever questioned why your results seemed so "wrong"? Well, in this workshop we'll be building our own admixture (ancestry) reports from publicly available data as well as showing how we can "hack race" to make racial results different.
* The respository is based on modified code from https://github.com/hagax8/ancestry_viz 

### Step 0: Download the Data
`chmod +x download_viz.sh`

`./download_viz.sh`

### Step 1: Run ancestry prediction and clustering
In this example, we're running a prediction on sample HG01108, a Puerto Rican female

http://www.internationalgenome.org/data-portal/sample/HG01108

`python2  hacker.py  --model GTM  --out classify_HG01108 --config ./standard_config.json  --classify-id HG01108`

### Step 2: Hack 3:D
For the ancestry hack, we will now shift her ancestry towards Kenyan using

`python2 hacker.py --model GTM --out kenya_dig_it --classify-id HG01108 --config ./standard_config.json --manipulate-towards "Luhya_in_Webuye,_Kenya"`

### Step 3: Check your work
* Now you should be able to check the output and cluster graphs :)





