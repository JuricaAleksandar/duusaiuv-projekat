# image.py - Primer detekcije objekata na jednoj slici koristeći YOLO v3 model
# (unapred istrenirani koeficijenti i konvertovani koeficijenti)

import os
import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import numpy as np
from yolov3 import YOLOv3Net

model_size = (416, 416, 3)          # Očekivani ulazni format za dati model i istrenirane koeficijente
num_classes = 80                    # Broj klasa nad kojima je mreža trenirana  
class_name = './data/coco.names'    # Putanja do datoteke koja sadrži imena klasa
max_output_size = 40                # Najveći broj okvira koje želimo da dobijemo za sve klase ukupno
max_output_size_per_class= 20       # Najveći broj okvira koje želimo da dobijemo po klasi
iou_threshold = 0.3                 # Prag mere preklapanja dva okvira
confidence_threshold = 0.75         # Prag pouzdanosti prisustva objekta 

cfgfile = './cfg/yolov3.cfg'                  # Putanja do YOLO v3 konfiguracione datoteke
weightfile = './weights/yolov3_weights.tf'    # Putanja do datoteke koja sadrži istrenirane koeficijente u TensorFlow formatu
left_img_path = "./DrivingStereo_demo_images/image_L"
right_img_path = "./DrivingStereo_demo_images/image_R"
output_path = './out/'

focal_length = 0.00754
camera_distance = 0.54
pitch = 0.00000586

h_threshold = 30
x_threshold = 80

sensor_vertical_pixels = 1200
sensor_horizontal_pixels = 1920


def preprocess_and_detect(model, img_path):     
    # Učitavanje ulazne slike i predobrada u format koji očekuje model
    img = cv2.imread(img_path)
    img = np.array(img)
    img = tf.expand_dims(img, 0) # Dodavanje još jedne dimenzije, jer je ulaz definsian kao batch slika
    resized_img = resize_image(img, (model_size[0], model_size[1]))

    # Inferencija nad ulaznom slikom
    # izlazne predikcije pred - skup vektora (10647), gde svaki odgovara jednom okviru lokacije objekta 
    pred = model.predict(resized_img)

    # Određivanje okvira oko detektovanih objekata (za određene pragove)
    boxes, scores, classes, nums = output_boxes( \
        pred, model_size,
        max_output_size=max_output_size,
        max_output_size_per_class=max_output_size_per_class,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold)

    # Izdvajanje pozicije i visine objekta
    obj_pos = []
    obj_height = []
    for box in np.array(boxes[0]):
        y = (box[3] + box[1])/2*sensor_vertical_pixels
        x = (box[2] + box[0])/2*sensor_horizontal_pixels
        h = (box[3] - box[1])*sensor_vertical_pixels
        obj_pos.append((x, y))
        obj_height.append(h)

    # Uklanja dimenzije veličine 1, npr. (28, 28, 3, 1) ->  (28, 28, 3)
    img = np.squeeze(img)

    return np.array(boxes[0]), np.array(scores[0]), np.array(classes[0]), np.array(nums[0]), img, obj_pos, obj_height


def try_match(c1ass, pos, height, candidate_class, candidate_pos, candidate_height, candidate_num):
    diff_y_list = []
    ind_list = []
    match_ind = None

    for candidate_ind in range(candidate_num):
        diff_y = abs(pos[1] - candidate_pos[candidate_ind][1])
        diff_h = abs(height - candidate_height[candidate_ind])
        diff_x = pos[0] - candidate_pos[candidate_ind][0]

        if c1ass == candidate_class[candidate_ind] and diff_h < h_threshold and 0 < diff_x < x_threshold:
            diff_y_list.append(diff_y)
            ind_list.append(candidate_ind)

    if diff_y_list:
        diff_y_min = min(diff_y_list)
        match_ind = ind_list[diff_y_list.index(diff_y_min)]

    return match_ind


def calculate_distance(left_obj_x, right_obj_x):
    pixel_dx = left_obj_x - right_obj_x
    dx = pixel_dx * pitch
    return camera_distance * focal_length / dx


def main():
    # Kreiranje modela
    model = YOLOv3Net(cfgfile, model_size, num_classes)

    # Učitavanje istreniranih koeficijenata u model
    model.load_weights(weightfile)

    # Učitavanje imena klasa
    class_names = load_class_names(class_name) 

    # Brisanje slika nastalih prilikome prethodnog pokretanja skripte
    old_images = os.listdir(output_path)
    for img in old_images:
        os.remove(os.path.join(output_path, img))

    img_names = os.listdir(left_img_path)
    for name in img_names:
        l_boxes, l_scores, l_classes, l_nums, img, l_obj_pos, l_obj_height = preprocess_and_detect(model, os.path.join(left_img_path, name))
        _, _, r_classes, r_nums, _, r_obj_pos, r_obj_height = preprocess_and_detect(model, os.path.join(right_img_path, name))

        valid_boxes = []
        valid_scores = []
        valid_classes = []
        valid_nums = 0
        distances = []

        if (l_nums > 0 and r_nums > 0):
            for l_obj_ind in range(l_nums):
                match_ind = try_match(c1ass=l_classes[l_obj_ind],
                                        pos=l_obj_pos[l_obj_ind],
                                        height=l_obj_height[l_obj_ind],
                                        candidate_class=r_classes,
                                        candidate_pos=r_obj_pos,
                                        candidate_height=r_obj_height,
                                        candidate_num=r_nums)

                if match_ind:
                    # Dodavanje u listu uparenih objekata
                    valid_boxes.append(l_boxes[l_obj_ind])
                    valid_scores.append(l_scores[l_obj_ind])
                    valid_classes.append(l_classes[l_obj_ind])
                    distances.append(calculate_distance(l_obj_pos[l_obj_ind][0], r_obj_pos[match_ind][0]))
                    valid_nums += 1

                    # Brisanje uparenog objekta iz liste kandidata
                    r_classes = np.delete(r_classes, match_ind, 0)
                    r_obj_pos = np.delete(r_obj_pos, match_ind, 0)
                    r_obj_height = np.delete(r_obj_height, match_ind, 0)
                    r_nums -= 1

        img = draw_outputs(img, valid_boxes, valid_scores, valid_classes, valid_nums, class_names, distances)

        # Čuvanje rezultata u datoteku
        cv2.imwrite(output_path + name[:-4] + '_OUT.png', img)


if __name__ == '__main__':
    main()