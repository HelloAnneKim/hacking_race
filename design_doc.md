# Project Planning
1. Reference Population retrieval
2. Replicate External Admixture Process
3. Similarity Function 
4. Single Target
5. Hide Target
6. Combos
7. Server

# Before Everything
- What language to code in?
- Project Name ?
- Github repo setup

## Reference Population Retrieval

### Resources
Sample reference populations: https://www.genome.gov/27528684/1000-genomes-project/
Sample reference populations: https://www.genome.gov/10001688/international-hapmap-project/

- total size of populations? size per sample? sample formats?
- Full genomes? just SNPs?

### Storage
- All in memory? What would RAM reqs be.
- Local Mongo or similar setup for lightweight local DB
- Postgres likely overkill unless setting up a custom server to host many populations
- Probably want to store the full samples


## Replicate External Admixture Process
- Clustering and defining the genetic commonality within each population genetically
 -- what does "defining the commonality within each population genetically" mean?
 -- Naive approach : For each SNP , run a clustering including SAMPLE and reference-pop-samples
 -- building the reverse of the function above seems difficult / not likely to be supported in a lib
- Enforce some constraints on Population Sizes (e.g. must have all labels, must have mutliple per label)


## Similarity Function 
- Given a SAMPLE, assign a score to each member of the reference-population based on similarity (shared SNPs)

## Target DEM
- Given a SAMPLE and TARGET_DEMOGRPAHIC, pick N most similar from TARGET_DEMOGRAPHIC and N most dissimilar from each other label. Run replicated_admixture_process

## Hide DEM
- Given a SAMPLE and TARGET_DEMOGRPAHIC, pick N most dissimilar from TARGET_DEMOGRAPHIC and N most similar from each other label. Run replicated_admixture_process

## Combos
- Allow for multiple Target DEMs and Hide DEMs (no overlap allowed)

## Server
- Are uploaded samples allowed? 
- 
- Is there an incremental clustering approach that could save work on the server-side?
- Hosting : Locally host for DEFCON demo? AWS?
- UI :
  -- Combos via Selectize
  -- Fancy Bar Chart breakdown a la Ancestry
  -- 'Fake' Ancestry / 23andMe Report generation?

