import cv2

# in_rtp = "udpsrc port=5200 ! application/x-rtp,\ encoding-name=JPEG,payload=26 ! rtpjpegdepay ! jpegdec ! autovideosink"
# cap = cv2.VideoCapture(in_rtp, cv2.CAP_GSTREAMER)

gstreamer_str='udpsrc port=8650 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, \
        encoding-name=(string)H264,\
         payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink'

cap = cv2.VideoCapture(gstreamer_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Cannot capture test src. Exiting.")
    quit()

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imshow("CVtest",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

