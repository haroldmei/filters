## Multiple Object Tracking with various Kalman Filter algorithm.    

Detected objects are shuffled, object id association is needed in order to keep track of a given object.   
Second order state is also supported so that object acceleration can be tracked (see 'circle-accel').   

For simplicity, the varying size of the object is not included in the state variable. In real world we also need to keep track of the change of object shape and object size.  

python show_detections.py -t circle  
python show_detections.py -t circle-accel    
python show_detections.py -t sliding  
python show_detections.py -t collide  


