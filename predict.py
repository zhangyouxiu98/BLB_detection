#----------------------------------------------------#
#   Single-image prediction, camera/video detection and
#   FPS testing are combined in one file; switch between
#   them with the `mode` variable.
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from deeplab import DeeplabV3

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   To change the color of a class, modify self.colors in __init__.
    #-------------------------------------------------------------------------#
    deeplab = DeeplabV3()
    #----------------------------------------------------------------------------------------------------------#
    #   mode selects the test mode:
    #   'predict'       predict a single image. To modify the prediction process (save image, crop object, etc.),
    #                   see the detailed comments below.
    #   'video'         video detection, using a camera or a video file. See the comments below.
    #   'fps'           test FPS, using street.jpg under img. See the comments below.
    #   'dir_predict'   iterate over a folder and save results. By default it iterates the img folder and
    #                   saves to img_out. See the comments below.
    #   'export_onnx'   export the model to onnx. Requires pytorch 1.7.1 or above.
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #-------------------------------------------------------------------------#
    #   count            whether to count the pixels (area) of the target and its ratio
    #   name_classes     the classes to distinguish, same as in json_to_dataset,
    #                    used to print class names and counts
    #
    #   count and name_classes only take effect when mode='predict'
    #-------------------------------------------------------------------------#
    count           = False
    # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    name_classes    = ["background","byk"]
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          path of the video; when video_path=0 it uses the camera.
    #                       To detect a video, set video_path = "xxx.mp4", which reads xxx.mp4 from the root.
    #   video_save_path     path to save the video; when video_save_path="" the video is not saved.
    #                       To save, set video_save_path = "yyy.mp4", saving yyy.mp4 to the root.
    #   video_fps           fps of the saved video.
    #
    #   video_path, video_save_path and video_fps only take effect when mode='video'.
    #   Saving a video requires ctrl+c, or running to the last frame, to finish writing.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       number of detections used when measuring fps. The larger it is, the more accurate.
    #   fps_image_path      image used for the fps test.
    #
    #   test_interval and fps_image_path only take effect when mode='fps'.
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     folder containing the images to detect.
    #   dir_save_path       folder where detected images are saved.
    #
    #   dir_origin_path and dir_save_path only take effect when mode='dir_predict'.
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/CUT/"
    dir_save_path   = "img_out/CUT/"
    #-------------------------------------------------------------------------#
    #   simplify            simplify the onnx model.
    #   onnx_save_path      path to save the onnx model.
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":
        '''
        A few notes about predict.py:
        1. This code cannot do batch prediction directly. To predict in batch, use
           os.listdir() to iterate over a folder and Image.open to open each file.
           See get_miou_prediction.py, which implements iteration.
        2. To save, use r_image.save("img.jpg").
        3. To avoid blending the original image with the segmentation, set blend to False.
        4. To extract a region from the mask, see the drawing part of detect_image:
           determine the class of each pixel, then extract the corresponding region.
        seg_img = np.zeros((np.shape(pr)[0],np.shape(pr)[1],3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c)*( self.colors[c][0] )).astype('uint8')
            seg_img[:, :, 1] += ((pr == c)*( self.colors[c][1] )).astype('uint8')
            seg_img[:, :, 2] += ((pr == c)*( self.colors[c][2] )).astype('uint8')
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = deeplab.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()

    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read the camera (video). Check whether the camera is installed (or the video path is correct).")

        fps = 0.0
        while(True):
            t1 = time.time()
            # Read a frame
            ref, frame = capture.read()
            if not ref:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Convert to Image
            frame = Image.fromarray(np.uint8(frame))
            # Run detection
            frame = np.array(deeplab.detect_image(frame))
            # Convert RGB to BGR for OpenCV display
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = deeplab.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = deeplab.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    elif mode == "export_onnx":
        deeplab.convert_to_onnx(simplify, onnx_save_path)

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
