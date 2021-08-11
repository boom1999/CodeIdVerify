# -*- coding: utf-8 -*-
# @Time : 2021/8/8 12:12
# @Author : lingz
# @Software: PyCharm

import random
import os
from PIL import Image, ImageDraw, ImageFont

# Set size for the image
width = 80
height = 20


def getRandomChar():
    """
    chr(i) return the Unicode of i
    :return: None
    """
    random_num = str(random.randint(0, 9))  # 0~9
    random_upper = chr(random.randint(65, 90))  # A~Z
    random_lower = chr(random.randint(97, 122))  # a~z
    random_char = random.choice([random_num, random_upper, random_lower])
    return random_char


def getRandomColor(is_light=True):
    """
    Generate colors, default light color
    :param is_light: Distinguish between light and dark colors
    :return: r, g, b
    """
    r = random.randint(0, 127) + int(is_light) * 128
    g = random.randint(0, 127) + int(is_light) * 128
    b = random.randint(0, 127) + int(is_light) * 128
    return r, g, b


def drawLine(draw):
    """
    Draw interference lines, numLine feels free
    x1,x2,y1,y2: The begin point and the end point of the line
    :param draw:
    :return: None
    """
    numLine = 5
    for i in range(numLine):
        x1 = random.randint(0, width)
        x2 = random.randint(0, width)
        y1 = random.randint(0, height)
        y2 = random.randint(0, height)
        draw.line((x1, y1, x2, y2), fill=getRandomColor(is_light=True))


def drawPoint(draw):
    """
    Draw interference points, numPoint feels free
    :param draw:
    :return: None
    """
    numPoint = 50
    for i in range(numPoint):
        x = random.randint(0, width)
        y = random.randint(0, height)
        draw.point((x, y), fill=getRandomColor(is_light=True))


def createImg(folder):
    """
    Use drawPoint(), drawLine(), getRandomColor(), getRandomChar() to generate an Img by random
    :param folder: train or test folder
    :return: None
    """
    # Generate background color by random
    bg_color = getRandomColor(is_light=True)
    # Generate a Img
    img = Image.new(mode="RGB", size=(width, height), color=bg_color)
    # Get picture brush
    draw = ImageDraw.Draw(img)
    # Set the font
    font = ImageFont.truetype(font="C:/Users/lingz/AppData/Local/Microsoft/Windows/Fonts/Monaco.ttf", size=18)
    # Set the Img name
    file_name = ''

    # Generate number or char, set the numOfNum to change the number of it
    numOfNum = 4
    for i in range(numOfNum):
        random_txt = getRandomChar()
        # Default: the bk_color is light, the txt_color is dark
        txt_color = getRandomColor(is_light=False)

        # Avoid the text color == the background color, re-generate txt_color
        while txt_color == bg_color:
            txt_color = getRandomColor(is_light=False)
        # Fill the txt, the first para is the coordinates of the upper left corner
        # If numOfNum = 4, then (15,0),(30,0),(45,0),(60,0), four nums of chars
        draw.text((15 + 15 * i, 0), text=random_txt, fill=txt_color, font=font)
        file_name += random_txt

    # Draw interference lines and points
    drawLine(draw)
    drawPoint(draw)
    print(file_name)

    # Save the Img
    with open("./{}/{}.png".format(folder, file_name), "wb") as train:
        img.save(train, format="png")


if __name__ == '__main__':
    # Generate num of Img
    num = 5000

    # Generate train and test folders
    os.path.exists('train') or os.makedirs('train')
    os.path.exists('test') or os.makedirs('test')

    for i in range(num):
        createImg('train')
        createImg('test')
