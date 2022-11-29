<div align="center">
  <p>
    <a align="center" href="https://ultralytics.com/yolov5" target="_blank">
      <img width="850" src="https://github.com/ultralytics/assets/raw/master/yolov5/v62/splash_readme.png"></a>
    <br><br>
    <a href="https://play.google.com/store/apps/details?id=com.ultralytics.ultralytics_app" style="text-decoration:none;">
      <img src="https://raw.githubusercontent.com/ultralytics/assets/master/app/google-play.svg" width="15%" alt="" /></a>&nbsp;
    <a href="https://apps.apple.com/xk/app/ultralytics/id1583935240" style="text-decoration:none;">
      <img src="https://raw.githubusercontent.com/ultralytics/assets/master/app/app-store.svg" width="15%" alt="" /></a>
  </p>
English | [简体中文](.github/README_cn.md)
<br>
<div>
   <a href="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml"><img src="https://github.com/ultralytics/yolov5/actions/workflows/ci-testing.yml/badge.svg" alt="CI CPU testing"></a>
   <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 Citation"></a>
   <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
   <br>
   <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
   <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
   <a href="https://join.slack.com/t/ultralytics/shared_invite/zt-w29ei8bp-jczz7QYUmDtgo6r6KcMIAg"><img src="https://img.shields.io/badge/Slack-Join_Forum-blue.svg?logo=slack" alt="Join Forum"></a>
</div>

<br>
<p>
YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com">Ultralytics</a>
 open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.
</p>

<div align="center">
   <a href="https://github.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-github.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.linkedin.com/company/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-linkedin.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://twitter.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-twitter.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.producthunt.com/@glenn_jocher">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-producthunt.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://youtube.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-youtube.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.facebook.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-facebook.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.instagram.com/ultralytics/">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-instagram.png" width="2%"/>
   </a>
</div>

<!--
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-api.png"></a>
-->

</div>

## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/pablomuo/yolov5_object_detection.git
cd Object_detection_using_yolov5
pip install -r requirements.txt
```

</details>

<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading [models](https://github.com/ultralytics/yolov5/tree/master/models) automatically from
the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          path/*.jpg  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

The commands below reproduce YOLOv5 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)
results. [Models](https://github.com/ultralytics/yolov5/tree/master/models)
and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases). Training times for YOLOv5n/s/m/l/x are
1/2/4/6/8 days on a V100 GPU ([Multi-GPU](https://github.com/ultralytics/yolov5/issues/475) times faster). Use the
largest `--batch-size` possible, or pass `--batch-size -1` for
YOLOv5 [AutoBatch](https://github.com/ultralytics/yolov5/pull/5092). Batch sizes shown for V100-16GB.

```bash
python train.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
                                       yolov5s                                64
                                       yolov5m                                40
                                       yolov5l                                24
                                       yolov5x                                16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

</details>

<details>
<summary>Tutorials</summary>

- [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)  🚀 RECOMMENDED
- [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)  ☘️
  RECOMMENDED
- [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)  🌟 NEW
- [Roboflow for Datasets, Labeling, and Active Learning](https://github.com/ultralytics/yolov5/issues/4975)  🌟 NEW
- [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
- [TFLite, ONNX, CoreML, TensorRT Export](https://github.com/ultralytics/yolov5/issues/251) 🚀

- [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
- [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
- [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
- [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)  ⭐ NEW
- [Architecture Summary](https://github.com/ultralytics/yolov5/issues/6998)  ⭐ NEW

</details>


<details>
<summary>Labeling</summary>

To label the database with the YOLOv5 format, the use of the ModifiedOpenLabelling tool is recommended. To install it and tag all images, follow the steps below:

```bash
# To install ModifiedOpenLabelling clone the following repository.
git clone https://github.com/ivangrov/ModifiedOpenLabelling
# In the file "class\_list.txt", list all classes to be detected.
# Paste all the images to be labeled in folder "images".
python run.py #execute the script
# Once all the images are finished, in the folder "bbox\_txt" will be a ".txt" file for each image.
```
</details>

## <div align="center">Training custom dataset</div>

- Once the whole database has been labelled, move the images for training to data\_train/images/train and the validation ones to data\_train/images/val. Move the labels of the training images to data\_train/labels/train, and the labels of validation to data\_train/labels/val.
- Go to folder "data" and modify the file "coco128.yaml". Firstly, indicate the total number of classes in variable "nc" and secondly, in variable "names" are the names of the classes, they must be in the same order in which they have been ordered when labelled.
- The model can already be trained. It is necessary to indicate the number of epochs, for example 300. Run the following code:
```bash
python train.py --data coco128.yaml --weights yolo5s.pt --epochs 300
```
- Once the model has finished the training, the weights will have been stored in runs/train/exp/weights/best.pt. To run the model with the weights of the training, run the following code:
```bash
python detect.py --weights runs/train/exp/weights/best.pt --source 0
```

## <div align="center">Running the pre-trained model</div>
For running the pre-trained model, the weights "weights.pt" must be used in the next code:
```bash
python detect.py --weights weights.pt --source 0
```

## <div align="center">Modifications to use Gstreamer and WebSocket</div>
<details>
<summary>Transmit with Gstreamer from AGV to Yolov5</summary>

To carry out the transmission of the stream from the AGV camera (Client I) to Yolov5 (Server I) is necessary the codes of the folder 'summit\_codes' where there are two files, 'vid\_tx.sh' and 'master\_edit.config' that are responsible for the transmission. In 'master\_edit.config the parameters are established and it will be necessary to indicate the 'IP' and the 'port' of the Server I. The code 'vid\_tx.sh' will will be in charge of making the transmission from the parameters of 'master\_edit.config'. In this code the parameters to transmit are set in:

```bash
IP_TX=10.236.24.39
PORT_TX=8554
```
</details>

<details>
<summary>Receive with Gstreamer from AGV </summary>

In order to (server) receive the streaming from the Robotnik Summit XL camera (Client I), it is necessary to modify the 'port' in the 'dataloaders.py' code through which it is transmitting the video.

```bash
class LoadStreams:
    def __init__():
    ...
       gstreamer_str = (f'udpsrc port={PORT_RX} auto-multicast=0 ! application/x-rtp, media=video, encoding-name=H264 !\
            rtpjitterbuffer latency=300 ! rtph264depay ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1')            
       cap = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)
    ...
```
</details>

<details>
<summary>Transmit with Gstreamer from Yolov5 to platform</summary>

To be able to transmit the video output of the YOLOv5 algorithm, it is necessary to set the 'IP' and the 'port' in the 'detect.py' code:

```bash
#Example to send the stream to the client II     
    #send the stream to the client via GStreamer
    gst_str_rtp =(f'appsrc ! videoconvert ! videoscale ! video/x-raw,format=I420,width=1280,height=720,framerate=20/1 !  videoconvert !\
         x264enc tune=zerolatency bitrate=3000 speed-preset=superfast ! rtph264pay !\
         udpsink host= {IP_TX} port= {PORT_TX}')
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out_send = cv2.VideoWriter(gst_str_rtp, fourcc, 20, (1280, 720), True)
```
</details>


<details>
<summary>Transmit with WebSocket</summary>

To be able to transmit the classes, weights, areas of the detected objects output of the YOLOv5 algorithm, it is necessary to set the 'IP' and the 'port' in the 'detect.py' code:

```bash
#send infor (text) via WebSocket to the Servidor_Broadcast and then to the platform 
    HOST_PORT = (f'ws://{IP_TX}:{PORT_SB}')
```
</details>

<details>
<summary>Transmit messages from Server II to the platform</summary>

In order to transmit the information from the server to the platform, it is necessary to establish the 'port' and the 'IP' of the server II, which is going to send the information to the platform (client II). This is set in 'servidor_broadcast.py' code:

```bash
   start_server = websockets.serve(echo, IP_TX, PORT_SB)
```
</details>

<details>
<summary>config.ini</summary>

The “config.ini” file is in charge of configuring the necessary specifications of “IP”, “port”, among others, which will be required at the time of the video transmission in real-time and the alert.
```bash
; config.ini
[DEFAULT]
PORT_RX= 8554
PORT_TX= 8650
IP_TX= 192.168.100.74
PORT_SB= 8000
```
</details> 

For a better understanding of the architecture, see the following figure

<div align="center">
  <p>
    <a align="center">
      <img width="850" src="https://github.com/pablomuo/yolov5_object_detection/blob/main/utils/Architecture.png"></a>
</div>

```bash

#To run the full yolov5 algorithm
python ejecutar.py
```

## <div align="center">YOLOv5 with Docker</div>
For running the algorithm with docker, as long as docker is downloaded, it is necessary to be located in the 'utils/docker' folder where the Dockerfile and the requirements.txt file are. Once there, the following code must be executed to create the yolov5 image:

```bash
docker build -t yolov5:v1 .
```

Once the image is created, it is necessary to execute the following code in the terminal, which will set the appropriate ’IP’ and ’port’ parameters and in turn, execute the yolov5 algorithm:

```bash
docker run -it --rm   --name yolo  --net=host  -e IP_TX=192.10.25.55   -e IP_SB=10.236.1.1   -e PORT_TX=8650   -e PORT_RX=8554   -e PORT_SB=8000   -e CONF_MIN=0.6 -e DISPLAY   -e QT_X11_NO_MITSHM=1   -v /tmp/.X11-unix:/tmp/.X11-unix   -v $HOME/.Xauthority:/root/.Xauthority   --device /dev/video0   --device /dev/video1 yolov5:v1
```



## <div align="center">Contribute</div>

We love your input! We want to make contributing to YOLOv5 as easy and transparent as possible. Please see our [Contributing Guide](CONTRIBUTING.md) to get started, and fill out the [YOLOv5 Survey](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey) to send us feedback on your experiences. Thank you to all our contributors!

<a href="https://github.com/ultralytics/yolov5/graphs/contributors"><img src="https://opencollective.com/ultralytics/contributors.svg?width=990" /></a>

## <div align="center">Contact</div>

For YOLOv5 bugs and feature requests please visit [GitHub Issues](https://github.com/ultralytics/yolov5/issues). For business inquiries or
professional support requests please visit [https://ultralytics.com/contact](https://ultralytics.com/contact).

<br>

<div align="center">
    <a href="https://github.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-github.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.linkedin.com/company/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-linkedin.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://twitter.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-twitter.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.producthunt.com/@glenn_jocher">
    <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-producthunt.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://youtube.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-youtube.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.facebook.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-facebook.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.instagram.com/ultralytics/">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-instagram.png" width="3%"/>
    </a>
</div>

[assets]: https://github.com/ultralytics/yolov5/releases
[tta]: https://github.com/ultralytics/yolov5/issues/303

