from PIL import Image, ImageDraw
import numpy as np
import math
import colorsys
import sys
import os
import re
import csv
import json

FPS = 30
PIXEL_PER_METER = 100
SUBMETER_PRECISION = 3
VERTICAL_ORIGIN_BOTTOM = 0
VERTICAL_ORIGIN_TOP = 1
VERTICAL_ORIGIN = VERTICAL_ORIGIN_BOTTOM
ACCEPT_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
IMAGE_SHAPE_INDEX_HEIGHT = 0
IMAGE_SHAPE_INDEX_WIDTH = 1
OBJECT_PATCH_SIZE = 4
OBJECT_DATA_LENGTH = 6 # x, y, R, G, B, A
OBJECT_INDEX_Y = 0
OBJECT_INDEX_X = 1
OBJECT_INDEX_R = 2
OBJECT_INDEX_G = 3
OBJECT_INDEX_B = 4
OBJECT_INDEX_A = 5
RECORD_PLACEHOLDER_MEMO = 'none'
PIXEL_DATA_LENGTH = 4 # R, G, B, A
PIXEL_CHANNEL_ALPHA = 3
MAX_ALPHA = 255


# object_move_records(python dict) 
# - 'key(hex-number)'
# -- record(python list) ... [time('00:00:00:00'), x(float), y(float)]
# -- record(python list) ... [time('00:00:00:00'), x(float), y(float)]
# -- ...                
object_move_records = dict()


def scan_image_files(folder):
    file_list = os.listdir(folder)
    file_list = sorted(file_list)
    for file in file_list:
        if(not file.endswith(ACCEPT_IMAGE_EXTENSIONS)):
            continue
        else:
            current_frame = int(file.split('.')[-2])

            file_path = os.path.join(folder, file)
            with Image.open(file_path) as image:
                scan_image(np.array(image), current_frame)

def scan_image(image, frame):
    # if VERTICAL_ORIGIN == VERTICAL_ORIGIN_BOTTOM:
    #     image = np.flipud(image)

    image_height = image.shape[IMAGE_SHAPE_INDEX_HEIGHT]
    image_width  = image.shape[IMAGE_SHAPE_INDEX_WIDTH]
    footprint_map = np.zeros((image_height, image_width))

    for y in range(0, image_height, OBJECT_PATCH_SIZE):
        for x in range(0, image_width, OBJECT_PATCH_SIZE):
            # 他オブジェクト認識時にすでに探索済ならスキップ
            if(footprint_map[y,x] == 1):
                continue
            
            # オブジェクト候補(透明でないピクセル)に突きあたったら、そこを仮の左上起点としてOBJECT_PATCH_SIZE四方のパッチがオブジェクト範囲になる
            pixel = image[y,x]
            if is_empty_pixel(pixel) == False:

                # y方向にOBJECT_PATCH_SIZE分飛ばしで走査しているので、より上(-y方向)にピクセルがあるかもしれない。上方向に移動して真の左上起点をみつける。
                for vy in range(y, y-OBJECT_PATCH_SIZE, -1):
                    pixel = image[vy-1,x]
                    if is_empty_pixel(pixel):
                        break
                
                # x方向も同じ
                for vx in range(x, x-OBJECT_PATCH_SIZE, -1):
                    pixel = image[vy, vx-1]
                    if is_empty_pixel(pixel):
                        break
                
                
                object = detect_object(image[vy:vy+OBJECT_PATCH_SIZE, vx:vx+OBJECT_PATCH_SIZE], vx, vy)
                object = object_pixel_to_meter(object, image_width, image_height)
                register_record(object, frame)
                
                footprint_map[vy:vy+OBJECT_PATCH_SIZE+1, x:vx+OBJECT_PATCH_SIZE+1] = 1
        

def detect_object(patch, left_x, top_y):
    object_center = np.zeros(2)
    object_total_weight = 0
    center_pixel = np.zeros(PIXEL_DATA_LENGTH)
    for y in range(OBJECT_PATCH_SIZE):
        for x in range(OBJECT_PATCH_SIZE):
            current_pixel = patch[y,x]
            alpha = current_pixel[PIXEL_CHANNEL_ALPHA]
            object_center[OBJECT_INDEX_Y] += y*alpha / MAX_ALPHA
            object_center[OBJECT_INDEX_X] += x*alpha / MAX_ALPHA
            object_total_weight += alpha / MAX_ALPHA

            # patchの中に必ずアンチエイリアスのかかっていないピクセルがあるので、そのピクセルのカラーを代表値として使う
            if alpha == MAX_ALPHA:
                center_pixel = current_pixel
    
    # 位置はピクセレートしなくてもいいので加重平均で取得する
    object_center /= object_total_weight

    object_center[OBJECT_INDEX_Y] += top_y
    object_center[OBJECT_INDEX_X] += left_x

    object = np.concatenate([object_center, center_pixel])
    return object

def register_record(object, frame):
    index_color_decimal = (int(object[OBJECT_INDEX_R]) << 16) + (int(object[OBJECT_INDEX_G]) << 8) + int(object[OBJECT_INDEX_B])
    index_hex_str = '{:x}'.format(index_color_decimal)
    time_str = frame_to_time(frame)

    record = [time_str, object[OBJECT_INDEX_X], object[OBJECT_INDEX_Y], RECORD_PLACEHOLDER_MEMO]
    
    if (index_hex_str in object_move_records) == False:
        object_move_records[index_hex_str] = []

    object_move_records[index_hex_str].append(record)
    # print(frame, record)
    return True

def is_empty_pixel(pixel):
    if(len(pixel) == 4):
        if(pixel[3] == 0 ):
            return True
    if(pixel[0] == pixel[1] == pixel[2] == 255):
        return True
    
    return False

def frame_to_time(frame):
    msec = math.floor(frame * 1000 / FPS)
    h = math.floor(msec / (1000 * 60 * 60))
    msec %= (1000 * 60 * 60)
    m = math.floor(msec / (1000 * 60))
    msec %= (1000 * 60)
    s = math.floor(msec / 1000)
    msec %= 1000
    msec = math.floor(msec / 10)

    time_str = '{:02d}:{:02d}:{:02d}:{:02d}'.format(h, m, s, msec)
    return time_str

def object_pixel_to_meter(object, image_width, image_height):
    pixel_x = object[OBJECT_INDEX_X]
    pixel_y = object[OBJECT_INDEX_Y]

    if VERTICAL_ORIGIN == VERTICAL_ORIGIN_BOTTOM:
        pixel_y = image_height - pixel_y

    pixel_x = pixel_x - image_width / 2

    object[OBJECT_INDEX_X] = pixel_x / PIXEL_PER_METER
    object[OBJECT_INDEX_Y] = pixel_y / PIXEL_PER_METER

    return object


def export_csv():
    for object_id, records in object_move_records.items():
        with open('out_' + object_id + '.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['time', 'x', 'y', 'misc'])
            writer.writerows(records)

    return True

def main():
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("folder", help="frame image files path", type=str)
    # argparser.add_argument("--csv", help="output as csv str", action="store_true")
    args = argparser.parse_args()
    if(os.path.exists(args.folder)):
        scan_image_files(args.folder)
        export_csv()


if __name__ == "__main__":
    main()
