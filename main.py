import os  # glob
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import io as IMGIO
from skimage.filters import sobel, threshold_otsu
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line, resize, rotate)
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.measure import label as conn
import numpy as np
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
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
    ax[2].set_xlim((0, img_gray.shape[1]))
    ax[2].set_ylim((img_gray.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()
    plt.savefig('edge_gray.png')
    plt.tight_layout()
    plt.show()

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


def repli_lines(line_exist, p0, p1):
    for l in line_exist:
        if l[0][0] == l[1][0] and p0[0] == p1[0]:
            if abs(l[0][1] - p0[1]) < 10:
                print('Remove Line! Exist Line: {}, Current Line: ({}, {})'.format(l, p0, p1))
                return True
        if l[0][1] == l[1][1] and p0[1] == p1[1]:
            if abs(l[0][0] - p0[0]) < 10:
                print('Remove Line! Exist Line: {}, Current Line: ({}, {})'.format(l, p0, p1))
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
    img_resized = resize(img, (img.shape[0] * 3, img.shape[1] * 3), anti_aliasing=True)
    if rotate_aug:
        img_pos = rotate(img_resized, PI/2 * Deg_2_Angle, resize=True)
        img_neg = rotate(img_resized, - PI/2 * Deg_2_Angle, resize=True)
        return [img_pos, img_neg, img_resized]
    else:
        return [None, None, img_resized]


def main(args):

    # load image first
    _, img_name = os.path.split(args.input_chart_path)
    assert CHART_TYPE[img_name[0]] == args.type, 'Input chart type different from the chart'
    try:
        IMG = Image.open(args.input_chart_path)
        img = np.asarray(IMG, dtype='uint8')  # 8 bit RGB
        img_gray = rgb2gray(img)  # range in [0,1] float64
        img_height, img_width = img_gray.shape
    except IOError:
        raise Exception(' Illegal chart file extension \n '
                        ' See PIL.image supporting format ')
    else:
        print(' Loaded chart from {}' .format(args.input_chart_path))

    # axes_detection
    axes_line = axes_detection(img_gray, img_height, img_width)
    print(' Axes location detected with location of {}'.format(axes_line))

    # OCR for printed text detection
    # run in bash: tesseract ~/gray.png stdout (# --tessdata-dir /n/home06/ywei1998/tesseract/share/tessdata)

    tool = pyocr.get_available_tools()[0]

    # left of y-axis and down of x-axis
    for line in axes_line:
        p0, p1 = line
        if p0[0] == p1[0]:
            # left of y-axis
            tmp_mask = np.zeros_like(img_gray)
            tmp_mask[:, :p0[0]] = 1
            img_left_y = img_gray * tmp_mask
            im = Image.fromarray((img_left_y * 255.0).astype('uint8'), mode='L')

            label_img, label_num = conn(input, neighbors=None, background=None, return_num=True, connectivity=None)
            # https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label

            im.save('./img_left_y.png')
            img_aug = ocr_image_preprocess(img_left_y, rotate_aug=True)
            txt_left_y = []
            for i in img_aug:
                if i is not None:
                    txt_left_y_cur = ocr(tool, i, 'txt')
                    txt_left_y.append(txt_left_y_cur)
                    print(" TXT for left from OCR {}" .format(txt_left_y_cur))
        if p0[1] == p1[1]:
            # down of x-axis
            tmp_mask = np.zeros_like(img_gray)
            tmp_mask[p0[1]:-1, :] = 1
            img_down_x = img_gray * tmp_mask
            im = Image.fromarray((img_down_x * 255.0).astype('uint8'), mode='L')
            im.save('./img_down_x.png')
            img_aug = ocr_image_preprocess(img_down_x, rotate_aug=True)
            txt_down_x = []
            for i in img_aug:
                if i is not None:
                    txt_down_x_cur = ocr(tool, i, 'txt')
                    txt_down_x.append(txt_down_x_cur)
                    print(" TXT for below from OCR {}" .format(txt_down_x_cur))


    # output:
    # TXT for below from OCR adAL
    # UBAILIW
    # uobep
    # Jep
    # suodg
    #
    # TXT for below from OCR Sports Car
    #
    # Type
    # TXT for below from OCR Sports Car
    #
    # Type
    # TXT for left from OCR uojes) Jad sai AiD
    # TXT for left from OCR City Miles Per Gallon
    # TXT for left from OCR uojes) Jad sai AiD

if __name__ == '__main__':
    args_input = parser.parse_args()
    main(args_input)
