# Rev_Vis

## Schedule

+ 29/07/2019
> Finish axes extraction  
> Try to use OCR under engine of [Tesseract](https://github.com/tesseract-ocr/tesseract) but not good  
> According the axes to crop subchart and resize&rotate chart for OCR, still not excellent  

+ 30/07/2019  
> Finish character extraction  
> Merge characters belonging to a word to bbox together  
> Run OCR on word image patch, and its augmentated counterparts (rotation and resize)  

+ 31/07/2019
> Vote on OCR output to get one specific result from all augmented image patches
> Finish axes attribution recover and naive text roles recover
> Finish plot data fetching from bar chart and save to json

+ 01/08/2019
> Finish support on line data extraction and dot data extraction  
> Debug previous bugs in axes and OCR and others..  

+ 07/08/2019
> Add in support for table-like chart data  

## Problem

+ 01/08/2019
> Axes recovery: seem to mix the right-y axis in bar chart and when have a dot on the axis, it will crash down..  
> OCR: OMG!! The OCR is so poor that my effort to extract each word into image patch is in vain and one more question is that when you can't give a good cluster number prior, it will run ocr on a multiword patch, and output like "40\n\n30". Should avoid this situation  
> Need to generalize to pie chart (without the axes)  

## Work of Perception Remains

+ 05/08/2019
> If you want a better line/eclipse/circle detection, you can try this: [ELSDc](https://github.com/viorik/ELSDc)  
> You need create a better text role classifier.  
> Fix all the bugs in previous section, and think deeply into you previous structure: how to organize it modularly and structurally.  

