
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from grabscreen import grab_screen
from PIL import Image
import cv2
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops



# ## Object detection imports
# Here are the imports from the object detection module.




from utils import label_map_util

from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


# ## Download Model



##
##opener = urllib.request.URLopener()
##opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
##tar_file = tarfile.open(MODEL_FILE)
##for file in tar_file.getmembers():
##  file_name = os.path.basename(file.name)
##  if 'frozen_inference_graph.pb' in file_name:
##    tar_file.extract(file, os.getcwd())
##

# ## Load a (frozen) Tensorflow model into memory.


####for allocating how much vram should process take
####gpu_options=tf.GPUOptions(per_prcoess_gpu_memory_fraction=0.70)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code

# In[8]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



        
        
                  


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    #with tf.Session(graph=graph,config=tf.ConfigProto(gpu_options=gpu_options))
    
    #https://www.tensorflow.org/performance/performance_guide#optimizing_for_gpu
    #intra_op_parallelism_threads: Setting this equal to the number of physical cores is recommended
    #inter_op_parallelism_threads: Setting this equal to the number of sockets is recommended.
    #OMP_NUM_THREADS: This defaults to the number of physical cores.

    #https://jhui.github.io/2017/03/07/TensorFlow-Perforamnce-and-advance-topics/
    #watch -n 2 nvidia-smi
    #with tf.device('/cpu:0'):

    #https://medium.com/@lisulimowicz/tensorflow-cpus-and-gpus-configuration-9c223436d4ef
      

    
    #with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1,intra_op_parallelism_threads=1,allow_soft_placement=True, device_count = {'CPU': 1})) as sess:
    with tf.Session() as sess:
      # Get handles to input and output tensors
      
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:        
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict               

while True:
  screen =cv2.resize(grab_screen(region=(0,0,800,600)),(800,450))
  image_np=cv2.cvtColor(screen ,cv2.COLOR_BGR2RGB)

  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(image_np,output_dict['detection_boxes'],output_dict['detection_classes'],output_dict['detection_scores'],category_index,instance_masks=output_dict.get('detection_masks'),use_normalized_coordinates=True,line_thickness=8)

  #warning if car is close
  for i,b in enumerate(output_dict['detection_boxes']):
    #3 is car and 5 is 6 and 8 is truck
    if output_dict['detection_classes'][i]==3 or output_dict['detection_classes'][i]==6 or output_dict['detection_classes'][i]==8:
      #0.5 min score to draw it set in vis_util
      if output_dict['detection_scores'][i]>0.5:    
        mid_x=(output_dict['detection_boxes'][i][3] + output_dict['detection_boxes'][i][1] )/ 2
        mid_y=(output_dict['detection_boxes'][i][2] + output_dict['detection_boxes'][i][0] )/2
        apx_distance=round((1-(output_dict['detection_boxes'][i][3] - output_dict['detection_boxes'][i][1]))**4,1)
        #cv2.putText(image,text,x_cord,y_cord,font_style,font_size,BGR_value,linewidth)
        cv2.putText(image_np,'{}'.format(apx_distance),(int(mid_x*800),int(mid_y*450)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        if apx_distance <=0.5:
          if mid_x > 0.3 and mid_x < 0.7:
            #BGR
            cv2.putText(image_np,'WARNING!!!',((int(mid_x*800)-50),(int(mid_y*450))),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),3)  
                    



  #displaying the output and destroying 
  cv2.imshow('GTAwindow',image_np)
  if cv2.waitKey(25) & 0xff== ord('q'):
    cv2.destroyAllWindows()
    break
  
                                  
                                  
                                  
                            
  
      
    
                        
                                  
                          
                                  
                                  
                                  
                                  


