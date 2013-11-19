Mojo is an interactive proofreading tool for annotation of 3D EM data.

Copyright (C) 2013 Harvard University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.


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