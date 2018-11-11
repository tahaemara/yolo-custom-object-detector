# Building a custom object detector using YOLOv3 in python

In this section, I put the code and the dataset of my rubik's cube to building a custom object detector to detect rubik's cube using YOLOv3 in Python. Training process is done using the Darknet framework and the real-time detector implemented with OpenCV DNN module.

  In the next table, I briefly described the contents of this section.

  <table class="table table-bordered table-striped" style="margin: 0 auto !important;float: none !important;width: auto;"> <thead> 									
	<tr> <td>Name</td> <td>Its Function</td> </tr> </thead>
	 <tbody> 
     	 <tr> <td>Dataset</td> <td>Folder contains images and labels folders.</td> </tr> 
<tr> <td>generate.py</td> <td>Script to generate train.txt and test.txt files.</td></tr>  
	 <tr> <td>custom</td> <td>Folder contains needed files (train.txt, test.txt, objects.names, yolov3-tiny.cfg, and trainer.data) for training. This folder must be pasted on the main directory of darknet</td> </tr> 
	<tr> <td>yolo_opencv.py</td> <td>Real-time rubik's cube detector, it reads a stream of frames from the webcam the then detects the rubik's cube in each one.</td> </tr> <tr> <td>model.data</td> <td>Pre-trained model to detect rubik's cube, can be downloaded from <a tyle="color:#337ab7;"    target="_blank" href="https://drive.google.com/file/d/1jBM9FzRSCVvOoBptUJSF51rvLJ_Tceu_/view?usp=sharing">here</a>.</td> </tr> 
</tbody></table>

<br><br>

### For more info

http://emaraic.com/blog/yolov3-custom-object-detector


### Video 

https://www.youtube.com/watch?v=tlVfsgRokcQ

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/tlVfsgRokcQ/3.jpg)](https://www.youtube.com/watch?v=tlVfsgRokcQ)
   
   
   

