# bandits
This repo includes the code for running some of the seminal papers on organizational learning under uncertainty.    

Currently, it includes the code for Denrell and March (2001) and Posen and Levinthal (2012).

There are thre veriants of Denrell and March (2001) the original is based on the learning and selection algortihms they show in their paper. The basic version is a simpler version based on softmax. The last version denrell_march is one that employs more flexible agents and bandits. 

There are two versions of Posen and Levinthal (2012), the original is the one used in the paper. The posen_levinthal version is more flexible with multiple learning and bandit generation fucntions

The next code to be included is Puranam and Swamy (2016).

# run online [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jparrieta/bandits/master)
You can run this code directly by opening the following binder in your browser or clicking the button above.
It takes a while to load. After loading, click any \*.ipynb  and you will be able to run an interactive verion of the Jupyter notebooks. You do not need to install Python or any dependencies, just run it from your browser.

**Link:** https://mybinder.org/v2/gh/jparrieta/bandits/master

# log  
**190916:** Added the original versions of Denrell and March (2001), before I used softmax, also added a simpler Posen and Levinthal (2012)
**190907:** Added a simplified version of Denrell and March (2001) with just uniform bandits and constant phi updating  
**190906:** Added Posen and Levinthal (2012), updated Denrell and March (2001), added e-greedy, greedy, and beta-distribution draws  
**190820:** Added Denrell and March (2001), with uniform, normal, and stable bandits, as well as constant phi and 1/k+1 modes  
