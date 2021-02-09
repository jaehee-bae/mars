# face parsing(segmentation)과 관련된 함수들
# 참고자료 : https://github.com/zllrunning/face-parsing.PyTorch

# face parsing
from model import BiSeNet
import torch
import torchvision.transforms as transforms

import cv2
import numpy as np
from PIL import Image

import utils as util

# << face parsing을 이용하여 헤어영역을 구하는 함수 >>
def find_hairarea(im, parsing_anno, stride):
    # PIL 이미지를 openCV 이미지로 변환
    im = util.piltocv(im)

    # 원본 얼굴 이미지 shape
    vis_im = im.copy().astype(np.uint8)

    # 부위 별 세그멘테이션 인덱스를 가지고 있는 2차원 배열
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    # 흰 배경에 검은 윤곽선을 저장하고 있는 이미지
    mask_img = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3), dtype=np.uint8)

    # 얼굴 부위 인덱스 최대 값
    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)

        # 부위영역 픽셀의 x, y 좌표 값
        y = index[0]
        x = index[1]

        if (len(x) <= 0) or (len(y) <= 0):  # 인덱스에 해당하는 부위영역이 없을 때
            if pi == 17:  # 헤어 영역에 해당하는 인덱스가 없을 때
                print("face feature (pi : ", pi, ") 을 찾을 수 없습니다. 다른 이미지를 이용하세요.")
                return

        else:
            if pi == 17:    # 헤어

                # cv2.imshow("vis_im", vis_im)
                # cv2.imshow("mask_img", mask_img)

                # 마스크 이미지에서 남길 영역의 픽셀만 원본 이미지 픽셀로 붙여넣기
                mask_img[y, x, :] = vis_im[y, x, :]

    return mask_img


# << 원본 이미지의 헤어영역을 검출하여 PIL 이미지로 반환해주는 함수 >>
def get_hair_area(src_img, img_size):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    # 세그멘테이션 모델 파일 호출
    net.load_state_dict(torch.load('79999_iter.pth'))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        # src_img를 PIL Image로 변경
        src_img = util.cvtopil(src_img)

        resize_img = src_img.resize((img_size, img_size), Image.BILINEAR)
        src_img = to_tensor(resize_img)
        src_img = torch.unsqueeze(src_img, 0)
        src_img = src_img.cuda()
        out = net(src_img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        # 헤어영역 검출
        hair_img = find_hairarea(resize_img, parsing, stride=1)

    return hair_img
