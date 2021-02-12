# image.py - Primer detekcije objekata na jednoj slici koristeći YOLO v3 model
# (unapred istrenirani koeficijenti i konvertovani koeficijenti)

import os
import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, draw_outputs_with_distance, resize_image
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

y_threshold = 20

def preprocess_and_detect(model, img_path):
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

    obj_pos = []
    obj_height = []
    for box in np.array(boxes[0]):
        y = (box[3] + box[1])/2*1200
        x = (box[2] + box[0])/2*1920
        h = (box[3] - box[1])*1200
        obj_pos.append((x, y))
        obj_height.append(h)

    return np.array(boxes[0]), np.array(scores[0]), np.array(classes[0]), np.array(nums[0]), img, obj_pos, obj_height


def main():
    # Kreiranje modela
    model = YOLOv3Net(cfgfile, model_size, num_classes)
    # Učitavanje istreniranih koeficijenata u model
    model.load_weights(weightfile)
    # Učitavanje imena klasa
    class_names = load_class_names(class_name)
    
    old_images = os.listdir(output_path)
    for img in old_images:
        os.remove(os.path.join(output_path, img))
    # Učitavanje ulazne slike i predobrada u format koji očekuje model
    img_names = os.listdir(left_img_path)
    for name in [img_names[120]]:
        boxes_l , scores_l, left_classes, nums_l, left_img, left_obj_pos, l_obj_height = preprocess_and_detect(model, os.path.join(left_img_path, name))
        boxes_r, scores_r, right_classes, nums_r, right_img, right_obj_pos, r_obj_height = preprocess_and_detect(model, os.path.join(right_img_path, name))
        print('\nDetektovano:')
        print(nums_l)
        print(nums_r)
        print(left_obj_pos[:nums_l])
        print(right_obj_pos[:nums_r])
        print(left_classes[:nums_l])
        print(right_classes[:nums_r]) 

        valid_boxes = []
        valid_scores = []
        valid_classes = []
        valid_nums = 0
        distances = []

        # Uklanja dimenzije veličine 1, npr. (28, 28, 3, 1) ->  (28, 28, 3)
        left_img = np.squeeze(left_img)
        right_img = np.squeeze(right_img)
        out_img = np.copy(left_img)

        left_img = draw_outputs(left_img, boxes_l, scores_l, left_classes, nums_l, class_names)
        right_img = draw_outputs(right_img, boxes_r, scores_r, right_classes, nums_r, class_names)

        if (nums_l > 0 and nums_r > 0):
            for l_obj_ind in range(nums_l):
                y_diff_list = []
                ind_list = []

                for r_obj_ind in range(nums_r):
                    diff_y = abs(left_obj_pos[l_obj_ind][1] - right_obj_pos[r_obj_ind][1])
                    diff_h = abs(l_obj_height[l_obj_ind] - r_obj_height[r_obj_ind])

                    if left_classes[l_obj_ind] == right_classes[r_obj_ind] and diff_y < y_threshold and diff_h < y_threshold:
                        y_diff_list.append(diff_y)
                        ind_list.append(r_obj_ind)

                print('Y DIFF LIST: {}'.format(y_diff_list))

                if y_diff_list:

                    min_y_diff = min(y_diff_list)
                    min_y_diff_ind = ind_list[y_diff_list.index(min_y_diff)]

                    pixel_dx = abs(left_obj_pos[l_obj_ind][0] - right_obj_pos[min_y_diff_ind][0])
                    dx = pixel_dx * pitch
                    distance = (camera_distance * focal_length) / dx

                    valid_boxes.append(boxes_l[l_obj_ind])
                    valid_scores.append(scores_l[l_obj_ind])
                    valid_classes.append(left_classes[l_obj_ind])
                    distances.append(distance)
                    valid_nums += 1

                    print('MATCHED: {}, {}'.format(l_obj_ind, min_y_diff_ind))
                    print(left_obj_pos[:nums_l])
                    print(right_obj_pos[:nums_r])
                    print(left_classes[:nums_l])
                    print(right_classes[:nums_r]) 
                    print('\n#########################################################\n')

                    right_classes = np.delete(right_classes, min_y_diff_ind, 0)
                    right_obj_pos = np.delete(right_obj_pos, min_y_diff_ind, 0)
                    r_obj_height = np.delete(r_obj_height, min_y_diff_ind, 0)
                    nums_r -= 1
                        

        print('\nValidno:')
        print(valid_boxes)
        print(valid_scores)
        print(valid_classes)
        print(valid_nums)
        print(distances)

        out_img = draw_outputs_with_distance(out_img, valid_boxes, valid_scores, valid_classes, valid_nums, class_names, distances)

        # Čuvanje rezultata u datoteku
        out_file_name = output_path + name
        cv2.imwrite(output_path + 'L_' + name, left_img)
        cv2.imwrite(output_path + 'R_' + name, right_img)
        cv2.imwrite(output_path + 'OUT_' + name, out_img)

if __name__ == '__main__':
    main()