## Tracking tech test sample code

Multiple Object Tracking using various Bayesian filters.

Shuffled the detected objects, object id association is supported.
Second order state is supported ('circle-accel').   

python show_detections.py -t circle  
python show_detections.py -t circle-accel    
python show_detections.py -t sliding  
python show_detections.py -t collide  

The `detections.py` file contains some example sets of detections that
you can use to test your tracking implementation.

The `show_detections.py` file is a program you can run to display these example
detections. It uses opencv to display images, so you may need to install
some system level dependencies. On Ubuntu you can use install the opencv dev
libs with `apt install libopencv-dev`, it should install all necessary dependencies.
