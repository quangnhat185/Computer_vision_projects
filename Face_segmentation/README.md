# Face segmentation with Grabcut (OpenCV)

## Demonstration
<p align="center"><img src="./demonstration.gif"></img></p>

## Examples 
__1. Segmenting face from image__
```
python face_segmentation.py -i {image_path}
```

__2. Segmenting face from video__
```
python face_segmentation.py -v {video_path} -o {output_path}
```

Example image and video can be found in folder `examples`.

## Optional argument
```
-h, --help            show this help message and exit
-i IMAGE, --image IMAGE
                      path to input image
-v VIDEO, --video VIDEO
                      path to input video
-f FACE, --face FACE  path to face detector model directory
-c ITER, --iter ITER  # of GrabCut iterations (larger value => slower
                      runtime)
-conf CONFIDENCE, --confidence CONFIDENCE
                      face detector filter threshold
-of OFFSET, --offset OFFSET
                      offset of boudning box around detected faces
-o OUTPUT, --output OUTPUT
                      path of processed ouput video
-s SHOW, --show SHOW  show video or not (y/n)

```
