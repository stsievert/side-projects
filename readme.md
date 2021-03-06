### Raspberry Pi Graphing Calculator
When I was doing math homework, I had my graphing calculator out, but I was solely using [Sage][sage] on my computer to do all of my derivatives and integrals. I asked myself, "why not simply have a hand held version of Sage?" To do that, I needed a Raspberry Pi, a screen, a keyboard, and three battery packs. Hooking all of these parts together was a significant process, as was getting the RPi to launch Sage on startup. The complete setup script, and related source files, can be found [here][rpi].

### MATLAB vs Python
You can best view the notebook at [nbviewer.ipython.org][nb]. This project does
speed tests between Matlab and Python, using Numpy and Numba v0.12.

###Innocentive Cancer Problem
Found on the [Innocentive website][innocentive], I was given the records of (approximately) 100 patients and 1,000 genes (of each patient) that had been tested for expression levels. Using the toolbox [r1magic][r1magic], I determined which [genes are expressed at greater levels in cancer patients.][result] Whether these genes cause cancer is a different problem. My solution took too long to complete on their servers, the reason I wasn't awarded the 10,000 dollar prize.
        
### Calculating powers
My goal here calculate [any number to the power of another number (e.g., 3<sup>3.5</sup>)][power], using only addition and multiplication (and the inverse operations). Though I'm certain there's faster, better algorithms out there, my solution was within [10^-6% after 10 iterations.][power-result] My solution required that I also write functions for natural logs.

### Project Euler
On Project Euler, you solve math problems with programming languages. The first problems are simple (1: what's the sum of every number below 1,000 divisible by 3 or 5?) to really complex (155: given 18 capacitors, how many possible capacitances can you have?). Total, I have [solved][euler-sol] 46 problems.


[innocentive]:https://www.innocentive.com
[r1magic]:https://github.com/msuzen/R1magic/wiki
[result]:https://github.com/scottsievert/side-projects/blob/master/InnoCentive_cancer_problem/correct!.png
[sage]:http://www.sagemath.org
[power]:https://github.com/scottsievert/side-projects/tree/master/Calculating_loqs/MATLAB_functions
[power-result]:https://github.com/scottsievert/side-projects/blob/master/Calculating_loqs/MATLAB_functions/S2pow3\%5E3.5.png
[euler]:http://projecteuler.net/about
[1]:http://projecteuler.net/problem=1
[nb]:http://nbviewer.ipython.org/urls/raw.github.com/scottsievert/side-projects/master/matlab_vs_python/Python%2520vs%2520Numba%2520vs%2520Matlab%2520vs%2520C.ipynb
[155]:http://projecteuler.net/problem=155
[rpi]:https://github.com/scottsievert/RPi
[euler-sol]:https://github.com/scottsievert/side-projects/blob/master/euler.py
