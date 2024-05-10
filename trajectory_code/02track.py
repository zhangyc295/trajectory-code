import sys

sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from moviepy.editor import *




def bbox_rel(*xyxy):
    """ Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    # bbox_right = max([xyxy[0].item(), xyxy[2].item()])
    # bbox_bottom = max([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

# input_file = './data/666Trim.MP4'
# out_file = './data/output.mp4'
#
# clip = VideoFileClip(input_file).resize(height=1080).set_fps(10)
# clip.write_videofile(out_file, codec='libx264')
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    # color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    # if save_vid or save_crop or show_vid:  # Add bbox to image
    #     c = int(cls)  # integer class 这是根据种类来划分颜色的
    #     id = int(id)
    #     label = f'{id} {names[c]} {conf:.2f}'
    #     # annotator.box_label(bboxes, label, color=colors(c, True))
    #     color = compute_color_for_labels(id)
    #     annotator.box_label(bboxes, label, color=color)
    color = (255, 255, 0)
    return tuple(color)

#
# def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
#     global frame_idx
#     for i, box in enumerate(bbox):
#         x1, y1, x2, y2 = [int(i) for i in box]
#         x1 += offset[0]
#         x2 += offset[0]
#         y1 += offset[1]
#         y2 += offset[1]
#         # box text and bar
#         id = int(identities[i]) if identities is not None else 0
#         color = compute_color_for_labels(id)
#         label = '{}{:d}'.format("", id)
#         t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
#         cv2.rectangle(
#             img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
#         cv2.putText(img, label, (x1, y1 +
#                                  t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
#     return img
def draw_boxes(img, bbox, class_labels,identities=None, offset=(0,0)):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        class_id = int(class_labels[i]) if class_labels is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}{}{:d}'.format("", id,'-',class_id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        # cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1), cv2.FONT_HERSHEY_PLAIN, 1, [255,255,255], 2)
    return img


# xyxy2tlwh函数  这个函数一般都会自带
def xyxy2tlwh(x):
    '''
    (top left x, top left y,width, height)
    '''
    y = torch.zeros_like(x) if isinstance(x,
                                          torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)



    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'
    # xlsx_path = str(Path(out)) + '/results.xlsx'
    dict_box = dict()

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []
                leibie = []
                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    if f'{names[int(cls)]}'=='nonvehicle':
                        leibie.append([1])
                    elif f'{names[int(cls)]}'=='vehicle':
                        leibie.append([2])
                    elif f'{names[int(cls)]}'=='bigvehicle':
                        leibie.append([3])
                    elif f'{names[int(cls)]}'=='truck':
                        leibie.append([4])
                    elif f'{names[int(cls)]}'=='bigtruck':
                        leibie.append([5])

                    # if f'{names[int(cls)]}'=='i2':
                    #     leibie.append([0])
                    # elif f'{names[int(cls)]}'=='i4':
                    #     leibie.append([1])
                    # elif f'{names[int(cls)]}'=='i5':
                    #     leibie.append([2])
                    # elif f'{names[int(cls)]}'=='il100':
                    #     leibie.append([3])
                    # elif f'{names[int(cls)]}'=='il60':
                    #     leibie.append([4])
                    # elif f'{names[int(cls)]}'=='il80':
                    #     leibie.append([5])
                    # elif f'{names[int(cls)]}'=='io':
                    #     leibie.append([6])
                    # elif f'{names[int(cls)]}'=='ip':
                    #     leibie.append([7])
                    # elif f'{names[int(cls)]}'=='p10':
                    #     leibie.append([8])
                    # elif f'{names[int(cls)]}'=='p11':
                    #     leibie.append([9])
                    # elif f'{names[int(cls)]}'=='p12':
                    #     leibie.append([10])
                    # elif f'{names[int(cls)]}'=='p19':
                    #     leibie.append([11])
                    # elif f'{names[int(cls)]}'=='p23':
                    #     leibie.append([12])
                    # elif f'{names[int(cls)]}'=='p26':
                    #     leibie.append([13])
                    # elif f'{names[int(cls)]}'=='p27':
                    #     leibie.append([14])
                    # elif f'{names[int(cls)]}'=='p3':
                    #     leibie.append([15])
                    # elif f'{names[int(cls)]}'=='p5':
                    #     leibie.append([16])
                    # elif f'{names[int(cls)]}'=='p6':
                    #     leibie.append([17])
                    # elif f'{names[int(cls)]}'=='pg':
                    #     leibie.append([18])
                    # elif f'{names[int(cls)]}'=='ph4':
                    #     leibie.append([19])
                    # elif f'{names[int(cls)]}'=='ph4.5':
                    #     leibie.append([20])
                    # elif f'{names[int(cls)]}'=='pl100':
                    #     leibie.append([21])
                    # elif f'{names[int(cls)]}'=='pl120':
                    #     leibie.append([22])
                    # elif f'{names[int(cls)]}'=='pl20':
                    #     leibie.append([23])
                    # elif f'{names[int(cls)]}'=='pl30':
                    #     leibie.append([24])
                    # elif f'{names[int(cls)]}'=='pl40':
                    #     leibie.append([25])
                    # elif f'{names[int(cls)]}'=='pl5':
                    #     leibie.append([26])
                    # elif f'{names[int(cls)]}'=='pl50':
                    #     leibie.append([27])
                    # elif f'{names[int(cls)]}'=='pl60':
                    #     leibie.append([28])
                    # elif f'{names[int(cls)]}'=='pl70':
                    #     leibie.append([29])
                    # elif f'{names[int(cls)]}'=='pl80':
                    #     leibie.append([30])
                    # elif f'{names[int(cls)]}'=='pm20':
                    #     leibie.append([31])
                    # elif f'{names[int(cls)]}'=='pm30':
                    #     leibie.append([32])
                    # elif f'{names[int(cls)]}'=='pm55':
                    #     leibie.append([33])
                    # elif f'{names[int(cls)]}'=='pn':
                    #     leibie.append([34])
                    # elif f'{names[int(cls)]}'=='pne':
                    #     leibie.append([35])
                    # elif f'{names[int(cls)]}'=='po':
                    #     leibie.append([36])
                    # elif f'{names[int(cls)]}'=='pr40':
                    #     leibie.append([37])
                    # elif f'{names[int(cls)]}'=='w13':
                    #     leibie.append([38])
                    # elif f'{names[int(cls)]}'=='w55':
                    #     leibie.append([39])
                    # elif f'{names[int(cls)]}'=='w57':
                    #     leibie.append([40])
                    # elif f'{names[int(cls)]}'=='w59':
                    #     leibie.append([41])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                leibies = torch.Tensor(leibie)

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss,leibies,im0)
                # print(len(outputs))
                # print(len(leibie))
                # leibie = np.array(leibie)
                # outputs = np.append(outputs,leibie,axis=1)
                # outputs = np.matrix.tolist(outputs)
                # for jj in range(len(bbox_xywh)):
                #     outputs[jj][0].append(leibie[jj][0])
                # outputs = np.array(outputs)

                # outputs = [x1, y1, x2, y2, track_id]
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]  # 提取前四列  坐标
                    identities = outputs[:, -1]  # 提取最后一列 ID
                    box_xywh = xyxy2tlwh(bbox_xyxy)
                    # xyxy2tlwh是坐标格式转换，从x1, y1, x2, y2转为top left x ,top left y, w, h 具体函数看文章最后
                    for j in range(len(box_xywh)):
                        x_center = box_xywh[j][0] + box_xywh[j][2] / 2  # 求框的中心x坐标
                        y_center = box_xywh[j][1] + box_xywh[j][3] / 2  # 求框的中心y坐标
                        id = outputs[j][-1]
                        center = [x_center, y_center]
                        dict_box.setdefault(id, []).append(center)  # 这个字典需要提前定义 dict_box = dict()

                    # COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0),
                    #              (210, 105, 30), (220, 20, 60),
                    #              (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237),
                    #              (138, 43, 226), (238, 130, 238),
                    #              (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0),
                    #              (255, 239, 213),
                    #              (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222),
                    #              (65, 105, 225), (173, 255, 47),
                    #              (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211),
                    #              (255, 99, 71), (144, 238, 144),
                    #              (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107),
                    #              (255, 255, 224), (128, 128, 128),
                    #              (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204),
                    #              (139, 69, 19), (255, 245, 238),
                    #              (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255),
                    #              (176, 224, 230), (0, 250, 154),
                    #              (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143),
                    #              (255, 0, 0), (240, 128, 128),
                    #              (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34),
                    #              (175, 238, 238), (255, 248, 220),
                    #              (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]
                    COLORS_10 = (188, 143, 143)

                    # #  计算帧率和
                    # indexIDs = []
                    # track_id_counter = 0
                    # for track in tracker.tracks:
                    #     if not track.is_confirmed() or track.time_since_update > 1:
                    #         continue
                    #
                    #     # print('track.track_id', track.track_id)
                    #
                    #     indexIDs.append(int(track.track_id))
                    #     counter.append(int(track.track_id))
                    #
                    # # 以下为画轨迹，原理就是将前后帧同ID的跟踪框中心坐标连接起来
                    # if frame_idx > 2:
                    #     for key, value in dict_box.items():
                    #         for a in range(len(value) - 1):
                    #             # color = COLORS_10[key % len(COLORS_10)]
                    #             color = COLORS_10
                    #             index_start = a
                    #             index_end = index_start + 1
                    #             cv2.line(im0, tuple(map(int, value[index_start])), tuple(map(int, value[index_end])),
                    #                      # map(int,"1234")转换为list[1,2,3,4]
                    #                      color, thickness=2, lineType=8)
                    # if len(dict_box) > 40:
                    #     keys_to_delete = list(dict_box.keys())[:len(dict_box) - 40]
                    #     for key in keys_to_delete:
                    #         del dict_box[key] # 最多存在的轨迹数

                # draw boxes for visualization
                if len(outputs) > 0:
                    # print(outputs)
                    # print(leibie)
                    # print(xywhs)
                    # print(confss)
                    # outputs = np.append(outputs,leibie,axis=1)
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    class_id = outputs[:, -2]
                    draw_boxes(im0, bbox_xyxy, class_id,identities)
                bbox_cen = 0
                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    targets = []
                    # car_class = []

                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        car_class = output[-2]
                        # with open(xlsx_path, 'a') as f:
                        #     f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                        #                                    bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format
                        a = bbox_left + (bbox_w - bbox_left) / 2
                        b = bbox_top + (bbox_h - bbox_top) / 2
                        # c = bbox_w - bbox_left + bbox_h - bbox_top
                        # """在这里定义一个k，用来确定车的类别，不同高度的的视频采用像素点确定不同的k
                        # h=60m时k取3.5/57，h=100m时k取3.5/42，h=180m时k取3.5/29
                        # 建议采用像素坐标确定.py确定"""
                        # k = 3.5 / 49
                        # if (bbox_w - bbox_left) * k + (bbox_h - bbox_top) * k < 4:
                        #     car_class = 0
                        # elif 4 <= (bbox_w - bbox_left) * k + (bbox_h - bbox_top) * k <= 11:
                        #     car_class = 1
                        # else:
                        #     car_class = 2
                        #
                        # if len(targets) > 0:
                        #     image_targets = targets[targets[:, 0] == i]
                        #     classes = image_targets[:, 1].astype('int')
                        #     cls = int(classes[j])
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 9 + '\n') % (identity, car_class, bbox_left,
                                                          bbox_top, bbox_w, bbox_h, a, b, frame_idx))  # label format
                        # with open(txt_path, 'a') as f:
                        #     f.write(('%g ' * 4 + '\n') % (frame_idx, identity,
                        #                                    x_center, y_center))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            # print('非机动车的框的大小为:', c)

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/1116best.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default=r'F:\66-82\N66-1.MP4', help='source')
    # parser.add_argument('--source', type=str,
    #                     default='./data/', help='source')
    # parser.add_argument('--fps', type=int, default=10, help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float,
    #                     default=0.65, help='object confidence threshold')   # 改的
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')  # 原来的
    # parser.add_argument('--iou-thres', type=float,
    #                     default=0.45, help='IOU threshold for NMS')  # 改的
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS') # 原来的
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0,2',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', default="true", action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    # parser.add_argument('--classes', nargs='+', type=int,
    #                     default=[0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41], help='filter by class')  # 0为机动车，1为非机动车，2为大车，跳帧操作在utils/datasets.py里
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0,1,2,3,4],
                        help='filter by class')  # 0为机动车，1为非机动车，2为大车，跳帧操作在utils/datasets.py里
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)