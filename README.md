<hr>  </hr>

<H1> Neural Network & CFD Swimmer Codes </H1>

<a href="https://github.com/nickabattista/FURM_ML_CFD_Swimmer"><img src="https://github.com/nickabattista/FURM_ML_CFD_Swimmer/blob/main/README_NN_Swimmer_Image.png" align="right" height="275" width="375" ></a>

Author: Nicholas A. Battista, Ph.D. <br>
Email: <a href="mailto:battistn[at]tcnj[.]edu"> battistn[at]tcnj.edu </a> <br>
Website: <a href="http://battistn.pages.tcnj.edu"> http://battistn.pages.tcnj.edu </a> <br>
Department: Mathematics & Statistics (<a href="https://mathstat.tcnj.edu/">TCNJ MATH</a>) <br>
Institution: The College of New Jersey (<a href="https://tcnj.edu/">TCNJ</a>) <br> 

<H4> Codes that accompany FURM Chapter "Using machine-learning and CFD to understand biomechanics of marine organisms" by NA Battista, AP Hoover, and MK Santiago. </H4>

<H4> Includes: </H4>

<ol type="1">
   <li> all codes to train, validate, and test a Neural Network, </li>
   <li> example codes for using the trained NN for performance prediction, and </li>
   <li> codes to perform a single swimmer simulation (leveraging the open-source immersed boundary software <a href="https://github.com/nickabattista/IB2d"> IB2d </a>) </li>
</ol>  

<hr>  </hr>


<H3> Video Tutorial for Visualizing Swimmer VTK Data w/ VisIt:</H3>

- Tutorial: <a href="https://youtu.be/4D4ruXbeCiQ"> https://youtu.be/4D4ruXbeCiQ </a>  
The basics of visualizing data using open source visualization software called <a href="https://wci.llnl.gov/simulation/computer-codes/visit/"> VisIt </a> (by Lawrence Livermore National Labs), visualizing the Lagrangian Points and Eulerian Data (colormaps for scalar data and vector fields for fluid velocity vectors)

<H4> VisIt Workaround for macOS 11 (Big Sur) and beyond </H4>

-->  Many VisIt users have gotten the following error when trying to open visualization files:

<p align="center"> "The MetaData server running on localhost could not get the file list for the current directory" </p>

--> To circumvent this error, try opening VisIt from the command line. Try the following:

<ol type="1">
  <li> In your terminal, change directory until you're in the directory you want to open (eg - the viz_IB2d folder) </li>
  <li> In your terminal, type: /Applications/VisIt.app/Contents/Resources/bin/visit -gui  (or the correct path based to where you installed VisIt) </li>
</ol>  

<hr> </hr>



