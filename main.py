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
                               probabilistic_hough_line, resize, rotate,
                               hough_circle, hough_circle_peaks)
from skimage.feature import canny, corner_harris, corner_peaks, corner_subpix
from skimage.color import (rgb2gray, label2rgb)
from skimage.measure import (label, regionprops)
from skimage.draw import circle_perimeter
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import pyocr
import pyocr.builders as ocrtools
import utils.utils as utils
import json
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


def lines_detection(img_gray, img_height, img_width):
    ## Sobel
    edge_sobel = sobel(img_gray)
    thresh = threshold_otsu(edge_sobel)
    edge_binary = edge_sobel > thresh
    # IMGIO.imsave('./edge.png', edge_sobel)
    # im = Image.fromarray((edge_binary * 255.0).astype('uint8'), mode='L')
    # im.save('./edge.png')

    lines = probabilistic_hough_line(edge_binary, threshold=50, line_length=50, #min(img_height, img_width)//2,
                                     line_gap=10, theta=np.asarray([-PI/2, 0, PI/2]))
    ## TODO: It seems that the dot on the axis will interrupt the line detection of image....

    # Save Result
    # fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    # ax = axes.ravel()
    #
    # ax[0].imshow(img_gray, cmap=cm.gray)
    # ax[0].set_title('Input image')
    #
    # ax[1].imshow(edge_binary, cmap=cm.gray)
    # ax[1].set_title('Sobel edges')
    #
    # ax[2].imshow(edge_binary * 0, cmap=cm.gray)
    line_exist = []
    for line in lines:
        p0, p1 = line
        # remove the replicated lines
        line_exist = repli_lines(line_exist, p0, p1)
        # 0 is x and 1 is y
    #         ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    # ax[2].set_xlim((0, edge_binary.shape[1]))
    # ax[2].set_ylim((edge_binary.shape[0], 0))
    # ax[2].set_title('Probabilistic Hough with replicated line remove')
    #
    # axes = axes_extraction(line_exist)
    # ax[3].imshow(edge_binary * 0, cmap=cm.gray)
    # ax[3].plot((axes[0][0][0], axes[0][1][0]), (axes[0][0][1], axes[0][1][1]))
    # ax[3].plot((axes[1][0][0], axes[1][1][0]), (axes[1][0][1], axes[1][1][1]))
    # ax[3].set_xlim((0, edge_binary.shape[1]))
    # ax[3].set_ylim((edge_binary.shape[0], 0))
    # ax[3].set_title('Axes Detection')
    #
    # for a in ax:
    #     a.set_axis_off()
    # plt.savefig('edge_gray.png')
    # plt.tight_layout()
    # plt.show()

    return line_exist

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


def axes_extraction(line_exist):
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
                    break
        if axes is not None:
            break
    return axes


def repli_lines(line_exist, p0, p1):
    add_flag = True
    for idx, l in enumerate(line_exist):
        if paral_line(l, (p0, p1)):
            if point_in_line_range(l, p0) and point_in_line_range(l, p1):
                # print('Remove Line! Exist Line: {}, Current Line: ({}, {})'.format(l, p0, p1))
                return line_exist
            # Need to replace?
            # line_exist[idx] = (p0, p1)

    if add_flag:
        line_exist.append((p0, p1))

    line_exist = list(dict.fromkeys(line_exist))
    return line_exist


def paral_line(line_1, line_2, thre=2e-2):
    line_1_p1, line_1_p2 = sorted(line_1)
    line_2_p1, line_2_p2 = sorted(line_2)
    if (line_1_p1[0] == line_1_p2[0]) and (line_2_p1[0] == line_2_p2[0]):
        return True
    elif (line_1_p1[0] == line_1_p2[0]) or (line_2_p1[0] == line_2_p2[0]):
        return False
    else:
        k_1 = (line_1_p2[1] - line_1_p1[1]) / (line_1_p2[0] - line_1_p1[0])
        k_2 = (line_2_p2[1] - line_2_p1[1]) / (line_2_p2[0] - line_2_p1[0])
        if abs(k_1 - k_2) < thre:
            return True
    return False


def point_in_line_range(l, p):
    if points_near(l[0], p, 10) or points_near(l[1], p, 10):
        return True
    return False


def points_near(p1, p2, dist=50):
    if (abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])) <= dist:
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
    img_resized_3 = resize(img, (img.shape[0] * 3, img.shape[1] * 3), anti_aliasing=True)
    img_resized_6 = resize(img, (img.shape[0] * 6, img.shape[1] * 6), anti_aliasing=True)
    if rotate_aug:
        img_pos_6 = rotate(img_resized_6, PI/2 * Deg_2_Angle, resize=True)
        img_neg_6 = rotate(img_resized_6, - PI/2 * Deg_2_Angle, resize=True)
        img_pos_3 = rotate(img_resized_3, PI/2 * Deg_2_Angle, resize=True)
        img_neg_3 = rotate(img_resized_3, - PI/2 * Deg_2_Angle, resize=True)
        return [img_pos_6, img_neg_6, img_resized_6, img_pos_3, img_neg_3, img_resized_3]
    else:
        return [None, None, img_resized_6, None, None, img_resized_3]


def ocr_postprocess(txt, loc, loc_bias, word_bbox_unexcept, role, len_begin):
    txt = txt.replace(',', '')  # 1,000->1000
    # if '\n' in txt:
    #     txt = txt.split('\n')
    #     txt = [word for word in txt if word != '']
    #     word_bbox_unexcept_len = len(word_bbox_unexcept)
    #     for word_idx, word in enumerate(txt):
    #         word_bbox_unexcept[word_bbox_unexcept_len+len_begin+word_idx] = {'bbox': loc+loc_bias,
    #                                                                         'txt': word,
    #                                                                         'role': role}
    #     return word_bbox_unexcept, txt, True
    # else:
    return word_bbox_unexcept, txt, False


def chart_ocr(img, word_bbox_role, tool, pad=False):
    word_bbox_except_len = len(word_bbox_role.values())
    word_bbox_unexcept = {}
    for idx, prop in word_bbox_role.items():
        minr, minc, maxr, maxc = prop['bbox']
        if pad:
            word_img = np.ones((2*(maxr - minr), 2*(maxc - minc)))
            word_img[(maxr - minr)//2:(maxr - minr)//2 + (maxr - minr),
                        (maxc - minc)//2:(maxc - minc)//2 + (maxc - minc)] = img[minr:maxr, minc:maxc]
        else:
            pad_size = 2
            word_img = img[minr-pad_size:maxr+pad_size, minc-pad_size:maxc+pad_size]

        # im = Image.fromarray((word_img * 255.0).astype('uint8'), mode='L')
        # im.save('./word_bbox_%d.png' % idx)

        img_aug = ocr_image_preprocess(word_img, rotate_aug=True)
        txt_cand = {}
        for idx_img, word_img_aug in enumerate(img_aug):
            if word_img_aug is not None:
                txt_tmp = ocr(tool, word_img_aug, 'txt')
                if txt_tmp != '':
                    if txt_tmp not in txt_cand:
                        txt_cand[txt_tmp] = 0
                    txt_cand[txt_tmp] += 1
        if not txt_cand:
            word_bbox_role[idx]['txt'] = 'UNKNOWN'
        else:
            ocr_voted = list(txt_cand.keys())[list(txt_cand.values()).index(sorted(txt_cand.values())[-1])]
            word_bbox_unexcept, ocr_voted, add_flag = \
                ocr_postprocess(ocr_voted, prop['bbox'], prop['bbox'], word_bbox_unexcept, prop['role'], word_bbox_except_len)
            if not add_flag:
                word_bbox_role[idx]['txt'] = ocr_voted

    if word_bbox_unexcept:
        word_bbox_role = utils.merge_dict(word_bbox_role, word_bbox_unexcept)

    return word_bbox_role


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
    # see https://scikit-image.org/docs/0.14.x/release_notes_and_installation.html#deprecations
    for region in regionprops(label_img, coordinates='xy'):  # remove warning in 0.14 vs 0.16
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


def character_cluster(chara_cent, chara_bbox, chara_bbox_area, img, axes):
    chara_cent = np.asarray(chara_cent)
    chara_bbox_area = np.asarray(chara_bbox_area)
    chara_bbox = np.asarray(chara_bbox)
    n_samples, n_features = chara_cent.shape
    # min_score = 1e10
    # kmeans_best = None
    efficiency = []
    inertia_last = 1e10
    n_digit_least = 12
    for n_digits in range(n_digit_least, min(15, n_samples)):
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

    print(efficiency)
    last_eff = list(filter(lambda x: x < 2/3, efficiency))[-1]
    n_digits = efficiency.index(last_eff) + n_digit_least
    print(' Pick digit number as: {}' .format(n_digits))
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

    # Naive set every words bbox with its role
    # TODO actually the role should defined not only by its own spatial location, but its relationship with other words
    # that is: the axis label of one axis is always in one line

    word_bbox_role = {}
    for idx, bbox in enumerate(word_bbox_postproc):
        p0, p1 = axes
        word_bbox_role[idx] = {'bbox': bbox}
        if (p0[0] == p1[0] and abs(bbox[3] - p0[0]) < 20) or \
            (p0[1] == p1[1] and abs(bbox[0] - p0[1]) < 20):  # min row and max column
            word_bbox_role[idx]['role'] = 'axis label'
        else:
            word_bbox_role[idx]['role'] = 'axis title'

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

    return word_bbox_role


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


def axis_map_recover(word_in_chart, axes):
    axis_prop = {'type': 'qualitative', 'quan_type': 'None', 'quan_map': []}

    p0, p1 = axes
    if p0[0] == p1[0]:
        axis_prop['loc'] = 'y'
    else:
        axis_prop['loc'] = 'x'
    axis_prop['loc_pixel'] = axes

    # recover axis type
    axis_word = {k: v for k, v in word_in_chart.items() if v['role'] == 'axis label'}
    axis_title = {v['txt']: v['bbox'].tolist() for k, v in word_in_chart.items() if v['role'] == 'axis title'}
    axis_known_value = []
    axis_known_loc = []
    for _, v in axis_word.items():
        txt = v['txt']
        bbox = v['bbox']
        if txt != 'UNKNOWN' and txt.isdigit():
            axis_prop['type'] = 'quantitative'
            if axis_prop['loc'] == 'y':
                axis_known_loc.append((bbox[0] + bbox[2]) / 2)  # row -> value
            else:
                axis_known_loc.append((bbox[1] + bbox[3]) / 2)  # column -> value
            axis_known_value.append(float(txt))

    # recover axis pixel to value map
    if axis_prop['type'] == 'quantitative':
        if len(axis_known_value) <= 1:
            print( ' \n Error during recover quantitative axis value: '
                   ' \n     because less than 2 axis values are detected by OCR!')
        else:
            if len(axis_known_value) == 2:
                print( ' \n Warning during recover quantitative axis value: '
                       ' \n     because only 2 axis values are detected by OCR'
                       ' \n     treat it as linear axis value here!')

            # take pixel_loc as x and txt as y -> function only support linear map for now
            LR_reg = LinearRegression().fit(np.asarray(axis_known_loc).reshape(-1, 1), np.asarray(axis_known_value))
            k = float(LR_reg.coef_)
            b = float(LR_reg.intercept_)
            axis_prop['quan_type'] = 'Linear'
            axis_prop['quan_map'] = [k, b]
            print(' Predict axis function as {} * x + {}'.format(k, b))

            # recover undetected axis label
            for idx, v in axis_word.items():
                txt = v['txt']
                bbox = v['bbox']
                if txt == 'UNKNOWN' or (not txt.isdigit()):
                    if axis_prop['loc'] == 'y':
                        loc = (bbox[0] + bbox[2]) / 2  # row -> value
                    else:
                        loc = (bbox[1] + bbox[3]) / 2  # column -> value
                    value = k * loc + b
                    axis_word[idx]['txt'] = str(int(round(value)))

    axis_word_tmp = axis_word
    axis_word = {v['txt']: v['bbox'].tolist() for k, v in axis_word_tmp.items()}

    axis_prop['title'] = axis_title
    axis_prop['label'] = axis_word

    return axis_prop


def bar_extraction(img_gray, axis_x, axis_y, img_height, img_width):
    ## Mask
    down_bound = axis_x[0][1]
    left_bound = axis_y[0][0]
    img = np.ones_like(img_gray)
    pad_size = 2
    img[:down_bound - pad_size, left_bound + pad_size:] = img_gray[:down_bound - pad_size, left_bound + pad_size:]

    ## Sobel
    edge_sobel = sobel(img)
    thresh = threshold_otsu(edge_sobel)
    edge_binary = edge_sobel > thresh

    lines = probabilistic_hough_line(edge_binary, threshold=50, line_length=10,
                                     line_gap=10, theta=np.asarray([-PI/2, PI/2]))  # only need horizontal ones

    line_exist = []
    for line in lines:
        p0, p1 = line
        # remove the replicated lines
        if repli_lines(line_exist, p0, p1) or abs(p0[1] - down_bound) < 10:
            continue
        else:
            line_exist.append(line)

    return line_exist


def bar_value_extraction(bar_lines, axis_x, axis_y):
    data_prop = {}
    if axis_y['quan_type'] == 'Linear':
        k = axis_y['quan_map'][0]
        b = axis_y['quan_map'][1]
    for axis_x_label, axis_x_label_loc in axis_x['label'].items():
        axis_x_label_cen = (axis_x_label_loc[1] + axis_x_label_loc[3])/2
        for bar_line in bar_lines:
            if min(bar_line[0][0], bar_line[1][0]) <= axis_x_label_cen <= max(bar_line[0][0], bar_line[1][0]):
                data_prop[axis_x_label] = k * bar_line[0][1] + b
                break
    return data_prop


def corner_extraction(img_bin, axis_x, axis_y):

    ## Mask
    down_bound = axis_x[0][1]
    left_bound = axis_y[0][0]
    img = np.zeros_like(img_bin)
    pad_size = 2
    img[:down_bound - pad_size, left_bound + pad_size:] = img_bin[:down_bound - pad_size, left_bound + pad_size:]

    corners = corner_peaks(corner_harris(img, k=0.05), min_distance=20)
    corners = corner_process(corners)
    coords_subpix = corner_subpix(img, corners, window_size=13)

    # fig, ax = plt.subplots(figsize=(10, 6), sharex=True, sharey=True)
    # ax.imshow(img, cmap=cm.gray)
    # ax.set_title('Corner Detection')
    # ax.plot(corners[:, 1], corners[:, 0], '.b', markersize=10)
    # ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
    # ax.set_axis_off()
    # plt.savefig('corner_plot.png')
    # plt.tight_layout()
    # plt.show()

    return corners


def corner_process(corners):
    # remove corner in the same line
    corners = sorted(corners[:, ::-1].tolist())  # sort with column value
    corner_post = []
    for corner in corners:
        if len(corner_post) < 2:
            corner_post.append(corner)
        else:
            cor_1 = corner_post[-1]
            cor_2 = corner_post[-2]
            if paral_line((cor_1, cor_2), (cor_1, corner), 5e-2) or paral_line((cor_1, cor_2), (cor_2, corner), 5e-2) \
                    or paral_line((cor_1, corner), (cor_2, corner), 5e-2):
                corner_post[-1] = corner
            else:
                corner_post.append(corner)

    return np.asarray(corner_post)[:, ::-1]


def line_value_extraction(corners, axis_x, axis_y):
    data_prop = {}
    if axis_y['quan_type'] == 'Linear':
        k = axis_y['quan_map'][0]
        b = axis_y['quan_map'][1]
    idx_begin = 0
    for axis_x_label, axis_x_label_loc in axis_x['label'].items():
        axis_x_label_cen = (axis_x_label_loc[1] + axis_x_label_loc[3])/2
        for idx in range(idx_begin, len(corners)):
            cor_l = corners[idx]
            cor_r = corners[idx+1]
            pad_size = 5
            if (cor_l[1] - pad_size) <= axis_x_label_cen < (cor_r[1] + pad_size):
                axis_x_label_y = (axis_x_label_cen - cor_l[1]) / (cor_r[1] - cor_l[1]) * (cor_r[0] - cor_l[0])
                axis_x_label_y += cor_l[0]
                data_prop[axis_x_label] = k * axis_x_label_y + b
                idx_begin = idx
                break
    return data_prop


def dot_detection(img_gray, axis_x, axis_y):
    ## Mask
    img_height, img_weight = img_gray.shape
    down_bound = axis_x[0][1]
    left_bound = axis_y[0][0]
    background = np.max(img_gray)
    img = np.zeros_like(img_gray)
    pad_size = 0
    img[:down_bound - pad_size, left_bound + pad_size:] = background - img_gray[:down_bound - pad_size, left_bound + pad_size:]

    thresh = threshold_otsu(img)
    img = img > thresh  # [False or True] bool

    label_img, label_num = label(img, neighbors=None, background=None, return_num=True, connectivity=None)
    circle_radius = []
    regions = regionprops(label_img, coordinates='xy')
    for region in regions:  # remove warning in 0.14 vs 0.16
        # take regions with large enough areas
        if region.bbox_area >= 20:
            circle_radius.append(region.equivalent_diameter//2)
    circle_r = max(set(circle_radius), key = circle_radius.count)
    print(' Most Radius: {}, and smallest radius: {}' .format(circle_r, sorted(circle_radius)[0]))
    label_centers = []
    idx = 0
    for region in regions:
        if abs(region.equivalent_diameter//2 - circle_r) > 1:
            label_patch = region.image
            label_patch = resize(label_patch, (label_patch.shape[0]*2, label_patch.shape[1]*2), anti_aliasing=True)
            # edge_sobel = sobel(label_patch)
            # thresh = threshold_otsu(edge_sobel)
            # edges = edge_sobel > thresh
            edges = canny(label_patch)
            hough_radii = np.asarray([circle_r * 2])  # np.arange(circle_r-1, circle_r+2)
            hough_res = hough_circle(edges, hough_radii)
            _, cx, cy, _ = hough_circle_peaks(hough_res, hough_radii, threshold=2/3 * np.max(hough_res))
            min_r, min_c, max_r, max_c = region.bbox
            cx = list(map(lambda x: x/2+min_c, cx))
            cy = list(map(lambda x: x/2+min_r, cy))

            for centroid in zip(cy, cx):
                label_centers.append(centroid)
        else:
            label_centers.append(region.centroid)
    print('{} Circles detected'.format(len(label_centers)))

    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 6))
    image_label_overlay = label2rgb(label_img, image=img)
    for label_center in label_centers:
        center_y, center_x = label_center
        circy, circx = circle_perimeter(int(center_y), int(center_x), int(circle_r), shape=[int(img_height), int(img_weight)])
        image_label_overlay[circy, circx] = (220, 20, 20)
    ax.set_title('Circle Detection')
    ax.imshow(image_label_overlay, cmap=plt.cm.gray)
    # plt.savefig('Circle_plot.png')
    plt.show()

    return label_centers


def dot_value_extraction(dots, axis_x, axis_y):
    data_prop = {}
    data_prop['Dot number'] = len(dots)
    if axis_y['quan_type'] == 'Linear':
        k_y = axis_y['quan_map'][0]
        b_y = axis_y['quan_map'][1]
    if axis_x['quan_type'] == 'Linear':
        k_x = axis_x['quan_map'][0]
        b_x = axis_x['quan_map'][1]

    for idx, dot in enumerate(dots):
        center_y, center_x = dot
        X = center_x * k_x + b_x
        Y = center_y * k_y + b_y
        data_prop[idx] = {'loc': dot, 'x_value': X, 'y_value': Y}
    return data_prop


def main(args):

    ### load image first
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

    ### axes_detection
    chart_line = lines_detection(img_gray, img_height, img_width)
    if args.type == 'dot':
        chart_line = [((136, 589), (136, 140)), ((136, 589), (684,589))]
    print(chart_line)

    axes_line = axes_extraction(chart_line)
    assert len(axes_line) >= 2, ' Error During Axes Detection. ' \
                                ' \n No two axes detected'
    axes_line = axes_line[:2]
    print(' Axes location detected with location of {}'.format(axes_line))

    ### OCR for printed text detection
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
            word_bbox_role = character_cluster(chara_cent, chara_bbox, chara_bbox_area, img_binary, line)
            word_in_chart = chart_ocr(img_gray, word_bbox_role, tool)
            print(word_in_chart)
            axis_prop_y = axis_map_recover(word_in_chart, line)
            print(axis_prop_y)

        if p0[1] == p1[1]:
            # down of x-axis
            img_down_x = np.zeros((img_binary.shape[0] - p0[1], img_binary.shape[1]), dtype='uint8')
            img_down_x = img_binary[p0[1]:, :]
            chara_cent, chara_bbox, chara_bbox_area = character_detection(img_down_x, 'conn_comp_down_x.png', [p0[1], 0])
            word_bbox_role = character_cluster(chara_cent, chara_bbox, chara_bbox_area, img_binary, line)
            word_in_chart = chart_ocr(img_gray, word_bbox_role, tool)
            print(word_in_chart)
            axis_prop_x = axis_map_recover(word_in_chart, line)
            print(axis_prop_x)
            # im = Image.fromarray((img_down_x * 255.0).astype('uint8'), mode='L')
            # im.save('./img_down_x.png')

    ### Plot data extraction
    if args.type == 'bar':
        ## Bar Chart
        bar_lines = bar_extraction(img_gray, axis_prop_x['loc_pixel'], axis_prop_y['loc_pixel'], img_height, img_width)
        data_prop = bar_value_extraction(bar_lines, axis_prop_x, axis_prop_y)

    if args.type == 'line':
        ## Line_Chart
        corners = corner_extraction(img_binary, axis_prop_x['loc_pixel'], axis_prop_y['loc_pixel'])
        data_prop = line_value_extraction(corners, axis_prop_x, axis_prop_y)

    if args.type == 'dot':
        dots = dot_detection(img_gray, axis_prop_x['loc_pixel'], axis_prop_y['loc_pixel'])
        data_prop = dot_value_extraction(dots, axis_prop_x, axis_prop_y)

    else:
        raise Exception(' \n Unsupported chart type input')

    # Merge dict: https://www.geeksforgeeks.org/python-merging-two-dictionaries/
    ### create chart json file
    chart_json = {'chart_name': img_name[:-4], 'chart_python': args.input_chart_path, 'chart_type': args.type,
                  'chart_data': data_prop, 'axis_1': axis_prop_x, 'axis_2': axis_prop_y}
    json_data = json.dumps(chart_json)
    with open('./%s_result.json' % img_name[:-4], 'w') as outfile:
        json.dump(chart_json, outfile, indent=4)

    print(' Write json from chart to {}' .format('./%s_result.json' % img_name[:-4]))


if __name__ == '__main__':
    args_input = parser.parse_args()
    main(args_input)
