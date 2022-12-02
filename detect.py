#enviamos el stream con Gstreamer y enviamos los datos con WEBSOCKETS
#enviamos la información de la clase cuando está 5 frames seguidos saliendo, más la información de control
#intentamos que salgo los boxes siempre y cuando en el frame anterior se ha detectado la clase, sino no, Para evitar que parpadee
#con el filtro modeo Pablo (pesos), ademas, con el filtro de area y vista de tiempo

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

#------------------------------------------------------------------------------------------------------------------------------------------------ import the corresponding libraries
import time
from imutils.video import VideoStream
import cv2
import asyncio
import websockets
import configparser
#------------------------------------------------------------------------------------------------------------------------------------------------

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

@torch.no_grad()
async def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    #-------------------------------------------------#To send stream and websocket to the client ---------------------------------------------------
    
    # To read the data infor of the client (IP, PORT)
    config = configparser.ConfigParser()
    config.read('config.ini')
    IP_TX = config['DEFAULT']['ip_tx']
    IP_SB = config['DEFAULT']['ip_sb']
    PORT_TX = config['DEFAULT']['port_tx']
    PORT_SB = config['DEFAULT']['port_sb']
    CONF_MIN = float(config['DEFAULT']['conf_min'])
    #AREA_MIN = float(config['DEFAULT']['AREA_MIN'])

    #send the stream to the client via GStreamer
    gst_str_rtp =(f'appsrc ! videoconvert ! videoscale ! video/x-raw,format=I420,width=1280,height=720,framerate=20/1 !  videoconvert !\
         x264enc tune=zerolatency bitrate=3000 speed-preset=superfast ! rtph264pay !\
         udpsink host= {IP_TX} port= {PORT_TX}')

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out_send = cv2.VideoWriter(gst_str_rtp, fourcc, 20, (1280, 720), True)  #out_send = cv2.VideoWriter(gst_str_rtp, fourcc, fps, (frame_width, frame_height), True) 

    #send infor (text) via WebSocket to the Servidor_Broadcast and then to the platform 
    HOST_PORT = (f'ws://{IP_SB}:{PORT_SB}')   

    #------------------------------------------------------------------------------------------------------------------------------------------------    #creamos vectores para almacenar la informacion
    final_class = [0]*26                                                                                           #se usa para llevar una cuenta de cuantos frames seguidos lleva viendo cada clase 
    eliminate = [0]*26                                                                                             #se utiliza saber cuando  poner a 0 el final_class por no aparecer en varios frames seguidos
    final_num = [0]*26                                                                                             #se usa para almacenar el numero de objetos que ve de cada una de las clases, 
    v_uni = [0]*26                                                                                                 #vector de 1 y 0s que se usa para multiplicar con el final_num para ver cuando sacar o no infor (si está)
    all_ima = [0]*26                                                                                               #se usa para llevar una cuenta de cuantos frames seguidos lleva viendo cada clase para ver si sacamos o no la img. 
    #------------------------------------------------------------------------------------------------------------------------------------------------   
        
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                #s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            #s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            time_det = time.time_ns()
            if len(det):
                
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                #------------------------------------------------------------------------------------------------------------------------------------------- creacion de listas que almacenan la infor de: 
                lista = []                                                                                          #almacena las clases
                lista_num = []                                                                                      #almacena los numeros de objetos que ve de las clases
                all_data = []                                                                                       #almacena el msg que queremos enviar al final (clase, peso, area...)
                area1 = []
                #-------------------------------------------------------------------------------------------------------------------------------------------
                for d in range(len(det)):                                                                           #area de cada una de las detecciones (oredenadas de mayor confianza a menos)
                    v = (det[d, 2] - det[d, 0]) * (det[d, 3] - det[d, 1])
                    area1.append(v)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()                                                                     # detections per class
                    for d in range(len(det)):                                                                       # number of detections per class
                        #if det[d, 5] == c and (det[d, 4] < CONF_MIN or area1[d] < AREA_MIN):                       #si se tiene en cuenta una confianza mínima y un área mínima
                        if det[d, 5] == c and det[d, 4] < CONF_MIN:                                                 #si se tiene en cuenta solo una confianza mínima 
                            n = n-1
                    
                    if n != 0:                                                                                      #toman valor las listas creadas anteriormente, cuando tengamos detecciones que complan las condiciones de peso y/o area
                        lista.append(int(c))
                        lista_num.append(int(n))
                ##------------------------------------------------------------------------------------------------------------------------------------------- analiza las clases que detectan en cada frame y suma 1 si se sale la clase en el frame
                for i1 in range(len(lista)):
                    numero = lista[i1] 
                    final_class[numero] += 1
                    final_num[numero] = lista_num[i1]
                    all_ima[numero] += 1
                ##------------------------------------------------------------------------------------------------------------------------------------------- 

                # Write results
                d = len(det)-1                                                                                      #contador invertido (va desde el final de det hasta el principio) (porque va desde los valores con menos confianza a los que más, justo lo contrario que det, que su primer valor es la detección con más confianza)
                for *xyxy, conf, cls in reversed(det):                                                              #este for va desde los valores con menos confianza hasta los que más tiene (lo inverso que det, que su primer valor es la detección con más confianza)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()                    # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)                                    # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:                                                           # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        ##------------------------------------------------------------------------------------------------------------------------------------------- #print the corresponding infor and bounding box according to the class and area
                        if all_ima[c] >= 2:
                            #if (det[d, 4] >= CONF_MIN) and (area1[d] >= AREA_MIN):                                 #si se tiene en cuenta una confianza mínima y un área mínima
                            if (det[d, 4] >= CONF_MIN):                                                             #si se tiene en cuenta una confianza mínima 
                                annotator.box_label(xyxy, label, color=colors(c, True))  
                                ms = f'{c:02d} {det[d, 4]:.4f} {det[d, 0]} {det[d, 1]} {det[d, 2]} {det[d, 3]} '    #estio sera lo que se imprima en el txt, de cada observacion
                                # print(ms)
                                save_data_text_old_version(ms)
                                msg_para_enviar = f'{c:02d} {det[d, 4]:.4f} {area1[d]}'                             #msg_para_enviar = f'{n:02d} {c:02d} {det[d, 4]:.4f} {area1[d]}'
                                all_data.append(msg_para_enviar)
                        d = d-1                                                                                     #se resta porque va desde el final de det hasta el principio (porque va desde los valores con menos confianza a los que más, justo lo contrario que det, que su primer valor es la detección con más confianza)
                        ##-------------------------------------------------------------------------------------------------------------------------------------------
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            ##------------------------------------------------------------------------------------------------------------------------------------------- 
            #este "for" ve si está la clase:
                #si no: le da valor a eliminarte y si este llega a ser igual a dos, inicializa las listas
                #si si, si final_class llega a ser multiplo de 5, se saca por pantalla, significa que lleva 5 frames seguidos saliendo esa clase
                #i3 selecciona el msg que corresponde a la clase del for i2
                #i4 se encarga de poner v_unit a 1 cuando se cumpla la condicion de >= 5, que permitira a b tomar valor y enviar el mensaje de control (msg_total)
                for i2 in range(len(final_class)):
                    if i2 in lista:
                        pass
                    else:
                        eliminate[i2] = eliminate[i2] +1

                    if eliminate[i2] == 2:
                        final_class[i2] = 0
                        eliminate[i2] = 0
                        all_ima[i2] = 0 
                    
                    if final_class[i2] == 0:
                        pass
                    elif (final_class[i2] % 5) == 0:                                                                 #evaluar si es multiplo de 5
                        final_class[i2] = 5

                        for i3 in range(len(all_data)):
                            num = f'{i2:02d}'
                            if all_data[i3][0:2] == num:        
                                msg_class = all_data[i3]
                                break
                        
                        # if len(all_data) != 0:
                        #     await enviar(msg_class, HOST_PORT)

                        for i4 in range(len(final_class)):
                            if final_class[i4] >= 5:
                                v_uni[i4] = 1
                        
                        b = [x*y for x,y in zip(final_num,v_uni)]                                                   #para multiplicar dos listas --> y ver si enviamos infor
                        msg_total = f"98 {b[0]:02d} {b[1]:02d} {b[2]:02d} {b[3]:02d} {b[4]:02d} {b[5]:02d} {b[6]:02d} {b[7]:02d} {b[8]:02d} {b[9]:02d} {b[10]:02d} {b[11]:02d} {b[12]:02d} {b[13]:02d} {b[14]:02d} {b[15]:02d} {b[16]:02d} {b[17]:02d} {b[18]:02d} {b[19]:02d} {b[20]:02d} {b[21]:02d} {b[22]:02d} {b[23]:02d} {b[24]:02d} {b[25]:02d}"
                        print(msg_total)
                        await enviar(msg_total, HOST_PORT)                       
                        v_uni = [0]*26

                        save_data_text(msg_total)
                    
        
                sum_data = sum(final_class)                                                                         #Este mensaje se enviara cuando este detectando clases, sin embargo, no se cumpla los requisitos del filtro(area-peso), por lo que no se tomaran en cuenta, para reestablecer el contador de la plataforma
                if sum_data == 0:
                    msg_empty = f"99"
                    await enviar(msg_empty, HOST_PORT)
                    
                    save_data_text(msg_empty)


            ##-------------------------------------------------------------------------------------------------------------------------------------------   #send msg if it is nothing in the streaming e inicializa los valores de conteo si pasan mas de dos frames seguidos sin ver las clases
            else: 
                lista = []                
                for i5 in range(len(final_class)):
                    eliminate[i5] = eliminate[i5] + 1
                    if eliminate[i5] == 2:
                        final_class[i5] = 0
                        eliminate[i5] = 0
                        all_ima[i5] = 0
                msg_para_enviar = f'99'
                await enviar(msg_para_enviar, HOST_PORT)
                
                save_data_text(msg_para_enviar)
            ##-------------------------------------------------------------------------------------------------------------------------------------------
            
            time_final = time.time_ns()
            # print(time_final)
            dif_time_total = (time_final-time_det)/1000000          # para tenerlo en ms
            # print("dif_time_total", dif_time_total)
            save_data(len(det),dif_time_total)
            # Stream results
            ##-------------------------------------------------------------------------------------------------------------------------------------------#send infor via gstreamer
            im0 = annotator.result()
            out_send.write(im0)
            ##-------------------------------------------------------------------------------------------------------------------------------------------
            if view_img:
                if p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0) #si la comentamos no retransmite la imagen
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)


    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

##------------------------------------------------------------------------------------------------------------------------------------------- funcion para enviar la informacion via websocket
async def enviar(msg_para_enviar, HOST_PORT):
    async with websockets.connect(HOST_PORT) as websocket:
        await websocket.send(msg_para_enviar)
##-------------------------------------------------------------------------------------------------------------------------------------------

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

async def main():
    opt = parse_opt()
    await run(**vars(opt))

def save_data(detections,lat):
    m, s = divmod(int(time.time() - start_time), 60)
    h, m = divmod(m, 60)
    if not os.path.exists("_results_latency.txt"):
        with open("_results_latency.txt", 'a') as outfile:
            # outfile.write("dif_time_class".rjust(8," ")\
            outfile.write("Detections".rjust(8," ")+ "   "+"Latency".rjust(8," ")\
            +"\n")
    with open("_results_latency.txt", 'a') as outfile:
        # outfile.write("{:8f}".format(x)\
        outfile.write(str(detections)+"   "+"{:08f}".format(lat)\
        +"   "+"\n")

def save_data_text(x):
    m, s = divmod(int(time.time() - start_time), 60)
    h, m = divmod(m, 60)
    if not os.path.exists("_results_text.txt"):
        with open("_results_text.txt", 'a') as outfile:
            outfile.write("Output".rjust(8," ")\
            # outfile.write("dif_time_class".rjust(8," ")+ "   "+"dif_time_total".rjust(8," ")\
            +"\n")
    with open("_results_text.txt", 'a') as outfile:
        outfile.write(x\
        # outfile.write("{:08f}".format(x)+"   "+"{:08f}".format(y)\
        +"   "+"\n")

def save_data_text_old_version(x):
    m, s = divmod(int(time.time() - start_time), 60)
    h, m = divmod(m, 60)
    if not os.path.exists("_results_text_old_version.txt"):
        with open("_results_text_old_version.txt", 'a') as outfile:
            outfile.write("Output_old_version".rjust(8," ")\
            # outfile.write("dif_time_class".rjust(8," ")+ "   "+"dif_time_total".rjust(8," ")\
            +"\n")
    with open("_results_text_old_version.txt", 'a') as outfile:
        outfile.write(x\
        # outfile.write("{:08f}".format(x)+"   "+"{:08f}".format(y)\
        +"   "+"\n")


if __name__ == "__main__":
    start_time         = time.time()
    check_requirements(exclude=('tensorboard', 'thop'))
    asyncio.run(main())
