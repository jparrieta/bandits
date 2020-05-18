# bandits
This repo includes the code for running some of the seminal papers on organizational learning under uncertainty.    

Currently, it includes the standaalone code for Denrell and March (2001), Posen and Levinthal (2012), Puranam and Swamy (2016), Denrell, Fang, and Levinthal (2004), and a full tutorial where the same agents and bandits replicate the first three papers.

The next step is to integrate Denrell, Fang, and Levinthal (2004) in the full tutorial. Or at least introduce a bit fo cleverness to the agents of that tutorial.  

# run online [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jparrieta/bandits/master)  
You can run this code directly by opening the following binder in your browser or clicking the button above.  
It takes a while to load. After loading, click any \*.ipynb  and you will be able to run an interactive verion of the Jupyter notebooks.   You do not need to install Python or any dependencies, just run it from your browser.  

**Link:** https://mybinder.org/v2/gh/jparrieta/bandits/master  

# log    

**200518:** Added Denrell, Fang, and Levinthal (2004). First Labyrinth!  
**191105:** Simplified the barebone version of Puranam and Swamy (2016).  
**191017:** Added standalone version of Puranam and Swamy (2016).  
**191016:** Added Puranam and Swamy (2016) to the full tutorial.  
**190927:** Fixed a bug in e-greedy and created a run_simulation function in the full tutorial.   
**190916:** Added a full_tutorial with all papers one after the other. Added direct replications of both papers.  
**190907:** Added a simplified version of Denrell and March (2001) with just uniform bandits and constant phi updating.    
**190906:** Added Posen and Levinthal (2012), updated Denrell and March (2001), added e-greedy, greedy, and beta-distribution draws.    
**190820:** Added Denrell and March (2001), with uniform, normal, and stable bandits, as well as constant phi and 1/k+1 modes.   
