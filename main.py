import os  # glob
import argparse
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from skimage import io as IMGIO
from skimage.filters import (sobel, threshold_otsu)
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line, resize, rotate)
from skimage.feature import canny
from skimage.color import (rgb2gray, label2rgb)
from skimage.measure import (label, regionprops)
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pyocr
import pyocr.builders as ocrtools
import utils.utils as utils
# import json
# import sys

CHART_TYPE = {
    'B': 'bar',
    'P': 'pie',
    'L': 'line',
    'T': 'table',
    'D': 'dot'
}
PI = 3.1415926
Deg_2_Angle = 180 / PI
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--type', default='bar',
                    choices=['bar', 'pie', 'line', 'table', 'dot'],
                    help='pick up input chart type')
parser.add_argument('-i', '--input_chart_path', default='./data/cropped/B_00.png', # required=True,
                    type=str, help='input chart path')


def axes_detection(img_gray, img_height, img_width):
    ## Sobel
    edge_sobel = sobel(img_gray)
    thresh = threshold_otsu(edge_sobel)
    edge_binary = edge_sobel > thresh
    # IMGIO.imsave('./edge.png', edge_sobel)
    # im = Image.fromarray((edge_binary * 255.0).astype('uint8'), mode='L')
    # im.save('./edge.png')

    lines = probabilistic_hough_line(edge_binary, threshold=50, line_length=min(img_height, img_width)//2,
                                     line_gap=10, theta=np.asarray([-PI/2, 0, PI/2]))

    # Save Result
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img_gray, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edge_binary, cmap=cm.gray)
    ax[1].set_title('Sobel edges')

    ax[2].imshow(edge_binary * 0, cmap=cm.gray)
    line_exist = []
    for line in lines:
        p0, p1 = line
        # remove the replicated lines
        if repli_lines(line_exist, p0, p1):
            continue
        else:
            line_exist.append(line)
            # 0 is x and 1 is y
            ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, edge_binary.shape[1]))
    ax[2].set_ylim((edge_binary.shape[0], 0))
    ax[2].set_title('Probabilistic Hough with replicated line remove')

    ax[3].imshow(edge_binary * 0, cmap=cm.gray)
    axes = []
    for line_1 in line_exist:
        for line_2 in line_exist:
            if line_1 == line_2:
                continue
            # first need vertical
            if (line_1[0][0] == line_1[1][0] and line_2[0][1] == line_2[1][1]) or \
                    (line_1[0][1] == line_1[1][1] and line_2[0][0] == line_2[1][0]):
                # second need to share a similar point
                if points_near(line_1[0], line_2[0]) or points_near(line_1[0], line_2[1]) or \
                        points_near(line_1[1], line_2[0]) or points_near(line_1[1], line_2[1]):
                    axes.append(line_1)
                    axes.append(line_2)
                    ax[3].plot((line_1[0][0], line_1[1][0]), (line_1[0][1], line_1[1][1]))
                    ax[3].plot((line_2[0][0], line_2[1][0]), (line_2[0][1], line_2[1][1]))
                    break
        if axes is not None:
            break
    ax[3].set_xlim((0, edge_binary.shape[1]))
    ax[3].set_ylim((edge_binary.shape[0], 0))
    ax[3].set_title('Axes Detection')

    for a in ax:
        a.set_axis_off()
    # plt.savefig('edge_gray.png')
    plt.tight_layout()
    plt.show()

    return axes

    '''
    ## Hough
    h, theta, d = hough_line(img_gray) #, np.asarray([-PI/2, 0, PI/2]))

    # Save Result
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(img_gray, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(img_gray, cmap=cm.gray)
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - img_gray.shape[1] * np.cos(angle)) / np.sin(angle)
        ax[1].plot((0, img_gray.shape[1]), (y0, y1), '-r')

    ax[1].set_xlim((0, img_gray.shape[1]))
    ax[1].set_ylim((img_gray.shape[0], 0))
    ax[1].set_title('Detected lines')

    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.savefig('edge_hough.png')
    plt.show()
    '''

    '''
    ## Canny
    edges = canny(img_gray, 3)# , 1, 25)
    lines = probabilistic_hough_line(edges, threshold=50, line_length=min(img_height, img_width)//2,
                                     line_gap=10, theta=np.asarray([-PI/2, 0, PI/2]))

    # Save Result
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img_gray, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edges')

    ax[2].imshow(edges * 0)
    line_exist = []
    for line in lines:
        p0, p1 = line
        # remove the replicated lines
        if repli_lines(line_exist, p0, p1):
            continue
        else:
            line_exist.append(line)
            ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, img_gray.shape[1]))
    ax[2].set_ylim((img_gray.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()
    plt.savefig('edge_canny.png')
    plt.tight_layout()
    plt.show()
    '''


def repli_lines(line_exist, p0, p1):
    for l in line_exist:
        if (l[0][0] == l[1][0] and p0[0] == p1[0]) or (l[0][1] == l[1][1] and p0[1] == p1[1]):
            if point_in_line_range(l, p0) and point_in_line_range(l, p1):
                print('Remove Line! Exist Line: {}, Current Line: ({}, {})'.format(l, p0, p1))
                return True
    return False


def point_in_line_range(l, p):
    if points_near(l[0], p, 10) or points_near(l[1], p, 10):
        return True
    return False


def points_near(p1, p2, dist=30):
    if (abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])) < dist:
        return True
    return False


def ocr(tool, img, cont='txt'):
    # img is [0,1] with float64 and single channel
    # langs = tool.get_available_languages()
    # lang = langs[0]

    if cont == 'txt':
        # txt is a Python string
        txt = tool.image_to_string(
            Image.fromarray((img * 255.0).astype('uint8'), mode='L'),
            lang="eng",
            builder=ocrtools.TextBuilder()
        )
        return txt

    if cont == 'word_boxes':
        # list of box objects. For each box object:
        #   box.content is the word in the box
        #   box.position is its position on the page (in pixels)
        #
        # Beware that some OCR tools (Tesseract for instance)
        # may return empty boxes
        word_boxes = tool.image_to_string(
            Image.fromarray((img * 255.0).astype('uint8'), mode='L'),
            lang="eng",
            builder=ocrtools.WordBoxBuilder()
        )
        return word_boxes

    if cont == 'line_word_boxes':
        # list of line objects. For each line object:
        #   line.word_boxes is a list of word boxes (the individual words in the line)
        #   line.content is the whole text of the line
        #   line.position is the position of the whole line on the page (in pixels)
        #
        # Beware that some OCR tools (Tesseract for instance)
        # may return empty boxes
        line_and_word_boxes = tool.image_to_string(
            Image.fromarray((img * 255.0).astype('uint8'), mode='L'),
            lang="eng",
            builder=ocrtools.LineBoxBuilder()
        )
        return line_and_word_boxes

    # if cont == 'digits':
    #     # Digits - Only Tesseract (not 'libtesseract' yet !)
    #     # digits is a python string
    #     digits = tool.image_to_string(
    #         Image.fromarray((img * 255.0).astype('uint8'), mode='L'),
    #         lang="eng",
    #         builder=pyocr.tesseract.DigitBuilder()
    #     )
    #     return digits

    else:
        raise ValueError(" Not supported OCR type ")


def ocr_image_preprocess(img, rotate_aug=True):
    img_resized = resize(img, (img.shape[0] * 4, img.shape[1] * 4), anti_aliasing=True)
    if rotate_aug:
        img_pos = rotate(img_resized, PI/2 * Deg_2_Angle, resize=True)
        img_neg = rotate(img_resized, - PI/2 * Deg_2_Angle, resize=True)
        return [img_pos, img_neg, img_resized]
    else:
        return [None, None, img_resized]


def chart_ocr(img, word_bbox, tool, pad=True):
    txt = {}
    for idx, bbox in enumerate(word_bbox):
        minr, minc, maxr, maxc = bbox
        if pad:
            word_img = np.ones((2*(maxr - minr), 2*(maxc - minc)))
            word_img[(maxr - minr)//2:(maxr - minr)//2 + (maxr - minr),
                        (maxc - minc)//2:(maxc - minc)//2 + (maxc - minc)] = img[minr:maxr, minc:maxc]
        else:
            word_img = img[minr:maxr, minc:maxc]

        # im = Image.fromarray((word_img * 255.0).astype('uint8'), mode='L')
        # im.save('./word_bbox_%d.png' % idx)

        img_aug = ocr_image_preprocess(word_img, rotate_aug=True)
        txt[idx] = {'bbox': bbox}
        for idx_img, word_img_aug in enumerate(img_aug):
            if word_img_aug is not None:
                txt[idx]['txt_%d' % idx_img] = ocr(tool, word_img_aug, 'txt')

        # TODO: Need to vote in all three orientation
    return txt


def character_detection(img, name, loc_bias=[0, 0]):
    # characters detection
    img_height, img_width = img.shape
    label_img, label_num = label(img, neighbors=None, background=None, return_num=True, connectivity=None)
    image_label_overlay = label2rgb(label_img, image=img)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    chara_bbox_cent = []
    chara_bbox = []
    chara_bbox_area = []
    for region in regionprops(label_img):
        # take regions with large enough areas
        if (img_height*img_width) // 9 >= region.bbox_area >= 20 \
                and (region.major_axis_length < 10.0 * region.minor_axis_length):
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=1)

            chara_bbox_cent.append(region.centroid)
            chara_bbox.append(region.bbox)
            chara_bbox_area.append(region.bbox_area)
            ax.add_patch(rect)

    plt.title(' Connection Components & Bboxes')
    ax.set_axis_off()
    # plt.savefig(name)
    plt.tight_layout()
    plt.show()

    # convert local center to global center
    from operator import add
    return list(map(lambda x: list(map(add, x, loc_bias)), chara_bbox_cent)), \
        list(map(lambda x: list(map(add, x, loc_bias + loc_bias)), chara_bbox)), \
        chara_bbox_area


def character_cluster(chara_cent, chara_bbox, chara_bbox_area, img):
    chara_cent = np.asarray(chara_cent)
    chara_bbox_area = np.asarray(chara_bbox_area)
    chara_bbox = np.asarray(chara_bbox)
    n_samples, n_features = chara_cent.shape
    # min_score = 1e10
    # kmeans_best = None
    efficiency = []
    inertia_last = 1e10
    for n_digits in range(1, 15):
        # print("n_digits: %d, \t n_samples %d, \t n_features %d" % (n_digits, n_samples, n_features))

        kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
        kmeans.fit(chara_cent, sample_weight=chara_bbox_area)

        # inertia_ Sum of squared distances of samples to their closest cluster center.
        # kmeans score: Opposite of the value of X on the K-means objective.
        kmeans_score = - kmeans.score(chara_cent)
        # print(' Score: {} Inertia: {}'.format(kmeans_score, kmeans.inertia_))
        efficiency.append(kmeans.inertia_ / inertia_last)
        inertia_last = kmeans.inertia_
        # if kmeans_score < min_score:
        #     min_score = kmeans_score
        #     kmeans_best = kmeans

    last_eff = list(filter(lambda x: x < 0.6, efficiency))[-1]
    n_digits = efficiency.index(last_eff) + 1
    print('Pick digit number as: {}' .format(n_digits))
    kmeans_best = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans_best.fit(chara_cent, sample_weight=chara_bbox_area)

    # merge bboxes which belong to one word together according to the bbox
    labels = kmeans_best.labels_
    word_bbox = {}
    for idx, label in enumerate(labels):
        bbox_add = chara_bbox[idx]
        if label not in word_bbox.keys():
            word_bbox[label] = bbox_add
        else:
            bbox_cur = word_bbox[label]
            bbox_cur = bbox_merge(bbox_cur, bbox_add)
            word_bbox[label] = bbox_cur

    # merge near word bounding box just in case of the non-perfect of KMeans:
    # and by the way, merge word in line together
    word_bbox_postproc = []
    for bbox_cur in word_bbox.values():
        minr_c, minc_c, maxr_c, maxc_c = bbox_cur
        if not word_bbox_postproc:
            word_bbox_postproc.append(bbox_cur)
            continue

        bbox_add_list = []
        for idx, bbox_exist in enumerate(word_bbox_postproc):
            minr_e, minc_e, maxr_e, maxc_e = bbox_exist
            if (line_overlap(minr_e, maxr_e, minr_c, maxr_c) and (abs(minc_c - maxc_e) < 10 or abs(maxc_c - minc_e) < 10)) \
                or (line_overlap(minc_e, maxc_e, minc_c, maxc_c) and (abs(minr_c - maxr_e) < 10 or abs(maxr_c - minr_e) < 10)):
                word_bbox_postproc.pop(idx)
                bbox_exist = bbox_merge(bbox_exist, bbox_cur)
                bbox_add_list.append(bbox_exist)

        if not bbox_add_list:
            bbox_add_list.append(bbox_cur)
        word_bbox_postproc += bbox_add_list

    print(' %d words bounding box detection' % len(word_bbox_postproc))
    plt.figure(1, figsize=(10, 6))
    plt.clf()
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()

    plt.imshow(img, cmap=cm.gray)

    plt.plot(chara_cent[:, 1], chara_cent[:, 0], 'b.', markersize=3)
    # Plot the centroids as a white X
    centroids = kmeans_best.cluster_centers_
    plt.scatter(centroids[:, 1], centroids[:, 0],
                marker='x', s=169, linewidths=3,
                color='g', zorder=10)

    for bbox_show in word_bbox_postproc:
        minr, minc, maxr, maxc = bbox_show
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.title('K-means clustering with words num of %d' % len(word_bbox_postproc))
    # plt.savefig('word_detection.png')
    plt.show()

    return word_bbox_postproc


def line_overlap(x1, y1, x2, y2, thre=None):
    max_x = max(x1, x2)
    min_x = min(x1, x2)
    max_y = max(y1, y2)
    min_y = min(y1, y2)
    max_len = max_y - min_x
    overlap = min_y - max_x
    if thre is None:
        min_len = min((y1-x1),(y2-x2))
        if overlap > min_len / 2:
            return True
    else:
        if overlap / max_len > thre:
            return True
    return False


def bbox_merge(bbox_cur, bbox_add):
    bbox_cur[0] = min(bbox_cur[0], bbox_add[0])
    bbox_cur[1] = min(bbox_cur[1], bbox_add[1])
    bbox_cur[2] = max(bbox_cur[2], bbox_add[2])
    bbox_cur[3] = max(bbox_cur[3], bbox_add[3])
    return bbox_cur


def main(args):

    # load image first
    _, img_name = os.path.split(args.input_chart_path)
    assert CHART_TYPE[img_name[0]] == args.type, 'Input chart type different from the chart'
    try:
        IMG = Image.open(args.input_chart_path)
        img = np.asarray(IMG, dtype='uint8')  # 8 bit RGB
        img_gray = rgb2gray(img)  # range in [0,1] float64
        thresh = threshold_otsu(img_gray)
        img_binary = img_gray < thresh  # [False or True] bool
        img_height, img_width = img_gray.shape
    except IOError:
        raise Exception(' Illegal chart file extension \n '
                        ' See PIL.image supporting format ')
    else:
        print(' Loaded chart from {}' .format(args.input_chart_path))

    # axes_detection
    axes_line = axes_detection(img_gray, img_height, img_width)
    axes_line = axes_line[:2]
    print(' Axes location detected with location of {}'.format(axes_line))

    # OCR for printed text detection
    # run in bash: tesseract ~/gray.png stdout (# --tessdata-dir /n/home06/ywei1998/tesseract/share/tessdata)

    tool = pyocr.get_available_tools()[0]

    # left of y-axis and down of x-axis
    for line in axes_line:
        p0, p1 = line
        if p0[0] == p1[0]:
            # left of y-axis
            img_left_y = np.zeros((img_binary.shape[0], p0[0]), dtype='uint8')
            img_left_y = img_binary[:, :p0[0]]

            chara_cent, chara_bbox, chara_bbox_area = character_detection(img_left_y, 'conn_comp_left_y.png', [0, 0])

            word_bbox = character_cluster(chara_cent, chara_bbox, chara_bbox_area, img_binary)

            word_in_chart = chart_ocr(img_gray, word_bbox, tool)
            print(" TXT for left from OCR {}".format(word_in_chart))

        if p0[1] == p1[1]:
            # down of x-axis
            img_down_x = np.zeros((img_binary.shape[0] - p0[1], img_binary.shape[1]), dtype='uint8')
            img_down_x = img_binary[p0[1]:-1, :]

            chara_cent, chara_bbox, chara_bbox_area = character_detection(img_down_x, 'conn_comp_down_x.png', [p0[1], 0])

            word_bbox =character_cluster(chara_cent, chara_bbox, chara_bbox_area, img_binary)

            word_in_chart = chart_ocr(img_gray, word_bbox, tool)
            print(" TXT for down from OCR {}".format(word_in_chart))

            # im = Image.fromarray((img_down_x * 255.0).astype('uint8'), mode='L')
            # im.save('./img_down_x.png')


if __name__ == '__main__':
    args_input = parser.parse_args()
    main(args_input)
