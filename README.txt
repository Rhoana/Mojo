Mojo is an interactive proof reading tool for annotation of 3D EM data.

    Copyright (C) 2013 Semour Knowles-Barley, Mike Roberts

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


The current version requires:

Windows 64-bit operating system

4GB RAM (8GB reccomended)

1GB GPU memory (1.5GB reccomended)

DirectX 11

(possibly more stuff - this is a beta test version)


Build Instructions:

The libraries in Mojo2.0\Mojo\Sdk is not included in the repository. Download from here:

http://people.seas.harvard.edu/~seymourkb/MojoSdk.zip

And unzip to Mojo2.0\Mojo\Sdk


Installation instructions:

1.  Copy the MojoRelease64 folder to your program files directory
    eg. C:\Program Files\MojoRelease64

2.  Install the Microsoft Visual Studio 2010 redistributable (64 bit)
    by running C:\Program Files\MojoRelease64\Install\vcredist_x64.exe
    
3.  Unzip the data ac3x75.zip training data to a working directory of you choice
    eg. ...\My Documents\ac3x75\         

4.  Run C:\Program Files\MojoRelease64\Mojo.Wpf.exe

    If there are errors at this point:
    
        Make sure you have DirectX 11 installed
        (run dxdiag.exe to determine the version you currently have)
    
        Check that you installed the Microsoft Visual Studio 2010
        redistributable (64 bit) (step 2)
        You might need to restart after this step
        
5.  File -> Load images...
    Select the mojo folder inside the training dataset directory
    eg.    ...\My Documents\ac3x75\mojo
    
    File -> Load Segmentation...
    Select the same mojo folder
    eg.    ...\My Documents\ac3x75\mojo
    
6.  Annotate some neurons (user guide pending...)
