# focusstack

Simple Focus Stacking in Python

This project implements a simple focus stacking algorithm in Python.

Focus stacking is useful when your depth of field is shallower than
all the objects you wish to capture (in macro photography, this is
very common).  Instead, you take a series of pictures where you focus
progressively throughout the range of images, so that you have a series
of images that, in aggregate, have everything in focus.  Focus stacking,
then, merges these images by taking the best focused region from all
of them to build up the final composite image.

The focus stacking logic is contained in FocusStack.py.  There is a
sample driver program in main.py.  It assumes that there is a subdirectory 
called "input" which contains the source images.  It generates an output 
called "merged.png".

I have also included some sample images in the input directory to allow
you to experiment with the code without having to shoot your own set of images.


This project is Copyright 2015 Charles McGuinness, and is released under the
Apache 2.0 license (see license file for precise details). 