#!/bin/bash

. master_edit.config
 
if [ $CODEC_TX == "JPEG" ]
then
  if [ $MODE == "VIDEO" ]
  then
    gst-launch-1.0 -v filesrc location = $VIDEO_NAME ! decodebin ! jpegenc quality=$QUALITY ! rtpjpegpay ! udpsink host=$IP_TX port=$PORT_TX sync=true
  elif [ $MODE == "WEBCAM" ]
  then
    gst-launch-1.0 -v v4l2src device=/dev/video$DEVICE ! video/x-raw,width=$WIDTH,height=$HEIGHT ! jpegenc quality=$QUALITY ! rtpjpegpay ! udpsink host=$IP_TX port=$PORT_TX sync=false
  fi
elif [ $CODEC_TX == "H264" ]
then
  if [ $MODE == "VIDEO" ]
  then
    echo "Video tx h264 not implemented"
    gst-launch-1.0 -v filesrc location = $VIDEO_NAME ! decodebin ! autovideoconvert ! x264enc name=videoEnc threads=$THREADS bitrate=$BITRATE tune=$TUNE speed-preset=$SPEED_PRESET key-int-max=$KEY_INT_MAX qp-min=$QP_MIN qp-max=$QP_MAX qp-step=$QP_STEP ! video/x-h264, profile=baseline ! rtph264pay config-interval=$CONFIG_INTERVAL mtu=$MTU name=$NAME ! udpsink host=$IP_TX port=$PORT_TX sync=true
  elif [ $MODE == "WEBCAM" ]
  then
    gst-launch-1.0 -v v4l2src device=/dev/video$DEVICE ! video/x-raw,width=$WIDTH,height=$HEIGHT ! autovideoconvert ! x264enc name=videoEnc threads=$THREADS bitrate=$BITRATE tune=$TUNE speed-preset=$SPEED_PRESET key-int-max=$KEY_INT_MAX qp-min=$QP_MIN qp-max=$QP_MAX qp-step=$QP_STEP ! video/x-h264, profile=baseline ! rtph264pay config-interval=$CONFIG_INTERVAL mtu=$MTU name=$NAME ! udpsink host=$IP_TX port=$PORT_TX sync=true 
  fi
fi






