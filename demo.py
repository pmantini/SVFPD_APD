import argparse
import glob
from peddla import peddla_net
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch, cv2
from video_cv import *
from tqdm import tqdm
import os, json

# def parse_args():
#
#     parser = argparse.ArgumentParser(description='Train SiamAF')
#     parser.add_argument('--img_list', type=str, default='files of image list')
#
#     args = parser.parse_args()
#     return args

def preprocess(image, mean, std):
    img = (image - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1)[np.newaxis, ...])

def parse_det(hm, wh, reg, density=None, diversity=None, score=0.1,down=4):
    # hm = _nms(hm, kernel=2)
    seman = hm[0, 0].cpu().numpy()
    height = wh[0, 0].cpu().numpy()
    offset_y = reg[0, 0, :, :].cpu().numpy()
    offset_x = reg[0, 1, :, :].cpu().numpy()
    density = density[0, 0].cpu().numpy()
    diversity = diversity[0].cpu().numpy()
    y_c, x_c = np.where(seman > score)
    maxh = int(down * seman.shape[0])
    maxw = int(down * seman.shape[1])
    boxs = []
    dens = []
    divers = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            w = 0.41 * h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, maxw), min(y1 + h, maxh), s])
            dens.append(density[y_c[i], x_c[i]])
            divers.append(diversity[:, y_c[i], x_c[i]])
        boxs = np.asarray(boxs, dtype=np.float32)
        dens = np.asarray(dens, dtype=np.float32)
        divers = np.asarray(divers, dtype=np.float32)
        keep = a_nms(boxs, 0.5, dens, divers)
        boxs = boxs[keep, :]
    else:
        boxs = np.asarray(boxs, dtype=np.float32)
    return boxs

def a_nms(dets, thresh, density, diversity):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        thresh_update = min(max(thresh, density[i]), 0.75)

        temp_tag = diversity[i]
        temp_tags = diversity[order[1:]]
        diff = np.sqrt(np.power((temp_tag - temp_tags), 2).sum(1))
        Flag_4 = diff > 0.95

        thresh_ = np.ones_like(ovr) * 0.5
        thresh_[Flag_4] = thresh_update
        inds = np.where(ovr <= thresh_)[0]
        order = order[inds + 1]

    return keep


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    return model

def main(file):
    input_file = file

    output_folder = input_file.rsplit("/", 1)[0].replace("SVFPD", "SVFPD_results")

    output_file = output_folder + "/apd.json"

    # BGR
    mean = np.array([0.485, 0.456, 0.406],
                    dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(3, 1, 1)


    num_layers = 34
    heads = {'hm': 1, 'wh': 1, 'reg': 2, 'aed': 4}
    model = peddla_net(num_layers, heads, head_conv=256, down_ratio=4).cuda().eval()

    # load model
    model = load_model(model, 'final.pth')
    # torch.cuda.empty_cache()

    video_sdv = SceneDatasetVideo(file)
    actual_size = video_sdv.get_frame_size()

    size = (1024, 2048)
    ratio = (float(actual_size[0])/float(size[1]), float(actual_size[1])/float(size[0]))

    dataloader_test = get_dataset(file, resizeTo=size, batch_size=1)
    result_detections = []

    for data in tqdm(dataloader_test):
        if data == None:
            break
        inputs, img_ids = data

        # inputs = np.swapaxes(np.swapaxes(inputs.numpy(), 1, 3), 1, 2)
        # inputs *= 255

        image_s = np.swapaxes(np.swapaxes(inputs.numpy(), 1, 3), 1, 2)[0]*255
        image_s = np.array(image_s, dtype=np.uint8)
        img = cv2.cvtColor(image_s, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(actual_size)
        img = cv2.resize(img, actual_size)

        torch.cuda.synchronize()
        # img = plt.imread(file).astype(np.float32)
        img_pre = inputs

        img_pre = (img_pre - mean) / std
        # img_pre = preprocess(img[:, :, ::-1], mean, std)
        img_pre = img_pre.cuda()

        with torch.no_grad():
            output = model(img_pre)[-1]

        output['hm'].sigmoid_()
        hm, wh, reg, attr = output['hm'], output['wh'], output['reg'], output['aed']

        density = attr.pow(2).sum(dim=1, keepdim=True).sqrt()

        diversity = torch.div(attr, density)
        boxes = parse_det(hm, wh, reg, density=density, diversity=diversity, score=0.5, down=4)

        for i in range(len(boxes)):
            x, y, w, h, score = boxes[i]
            tmp = {}

            tmp['image_id'] = int(img_ids.numpy()[0] + 1)
            tmp['category_id'] = int(0)
            tl_x = x
            tl_y = y

            br_x = w
            br_y = h
            # tmp['bbox'] = [(int((k[0]-k[2]/2)*actual_size[0]), int((k[1]-k[3]/2)*actual_size[1])), (int((k[0]+k[2]/2)*actual_size[0]), int((k[1]+k[3]/2)*actual_size[1]))]
            # print([int(tl_x), int(tl_y),
            #                int(br_x), int(br_y)], size, ratio)
            bbox = [int(tl_x*ratio[0]), int(tl_y*ratio[1]),
                           int(br_x*ratio[0]), int(br_y*ratio[1])]

            tmp['bbox'] = [bbox[1], bbox[0],
                           bbox[3], bbox[2]]

            tmp['score'] = float(score)
            result_detections.append(tmp)
            # img_gh = cv2.cvtColor(image_s, cv2.COLOR_RGB2GRAY)

            # img = cv2.rectangle(img, (tmp['bbox'][0], tmp['bbox'][1]), (tmp['bbox'][2], tmp['bbox'][3]), 255, 2)
        # print(type(image_s))
        # image_s = np.array(image_s, dtype=np.uint8)
        # cv2.imshow("image", img)
        # cv2.waitKey(1)


    try:
        os.makedirs(output_folder)
    except:
        print("Folder exists!")

    with open(output_file, 'w') as f:
        json.dump(result_detections, f)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("-f", "--file", dest="file",
                        help="specify name of the file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="output",
                        help="specify name of the output", metavar="OUTPUT")

    args = parser.parse_args()

    main(args.file)