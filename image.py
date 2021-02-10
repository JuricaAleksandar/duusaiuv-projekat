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
confidence_threshold = 0.7          # Prag pouzdanosti prisustva objekta 

cfgfile = './cfg/yolov3.cfg'                  # Putanja do YOLO v3 konfiguracione datoteke
weightfile = './weights/yolov3_weights.tf'    # Putanja do datoteke koja sadrži istrenirane koeficijente u TensorFlow formatu
left_img_path = "./DrivingStereo_demo_images/image_L"
right_img_path = "./DrivingStereo_demo_images/image_R"

focal_length = 0.00754
camera_distance = 0.54
pitch = 0.00000586

def pretprocessing(model, img_path):
    img = cv2.imread(img_path)
    img = np.array(img)
    img = tf.expand_dims(img, 0) # Dodavanje još jedne dimenzije, jer je ulaz definsian kao batch slika
    resized_img = resize_image(img, (model_size[0],model_size[1]))

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
    list_of_x = list()
    for box in np.array(boxes[0]):
        x = (box[2] + box[0])/2
        if x != 0:
            list_of_x.append(x)
    return boxes, scores, classes, nums, img, list_of_x


def main():
    # Kreiranje modela
    model = YOLOv3Net(cfgfile, model_size, num_classes)
    # Učitavanje istreniranih koeficijenata u model
    model.load_weights(weightfile)
    # Učitavanje imena klasa
    class_names = load_class_names(class_name)
    
    # Učitavanje ulazne slike i predobrada u format koji očekuje model
    img_names = os.listdir(left_img_path)
    for path_to_image in img_names:
        boxes_l , scores_l, left_classes, nums_l, left_img, list_of_left_x = pretprocessing(model, os.path.join(left_img_path, path_to_image))
        boxes_r, scores_r, right_classes, nums_r, right_img, list_of_right_x = pretprocessing(model, os.path.join(right_img_path, path_to_image))
        if list_of_left_x and list_of_right_x:
            # Uklanja dimenzije veličine 1, npr. (28, 28, 3, 1) ->  (28, 28, 3)
            left_image = np.squeeze(left_img) 
            out_img_l = draw_outputs(left_image, boxes_l, scores_l, left_classes, nums_l, class_names)

            # Uklanja dimenzije veličine 1, npr. (28, 28, 3, 1) ->  (28, 28, 3)
            right_image = np.squeeze(right_img) 
            out_img_r = draw_outputs(right_image, boxes_r, scores_r, right_classes, nums_r, class_names)

            # Čuvanje rezultata u datoteku
            out_file_name_l = './out/{}L.png'.format(path_to_image[:-4])
            cv2.imwrite(out_file_name_l, out_img_l)

            # Čuvanje rezultata u datoteku
            out_file_name_r = './out/{}R.png'.format(path_to_image[:-4])
            cv2.imwrite(out_file_name_r, out_img_r)

            # left_image = cv2.imread(os.path.join(right_img_path, right_img_names[4]))
            # left_image = np.array(left_image)
            # left_image = tf.expand_dims(left_image, 0) # Dodavanje još jedne dimenzije, jer je ulaz definsian kao batch slika
            # resized_left_image = resize_image(left_image, (model_size[0],model_size[1]))

            # # Inferencija nad ulaznom slikom
            # # izlazne predikcije pred - skup vektora (10647), gde svaki odgovara jednom okviru lokacije objekta 
            # pred = model.predict(resized_left_image)

            # # Određivanje okvira oko detektovanih objekata (za određene pragove)
            # boxes, scores, classes, nums = output_boxes( \
            #     pred, model_size,
            #     max_output_size=max_output_size,
            #     max_output_size_per_class=max_output_size_per_class,
            #     iou_threshold=iou_threshold,
            #     confidence_threshold=confidence_threshold)
            
            min_num_classes = min(nums_l[0], nums_r[0])
            # caci = nums_l[0] < nums_r[0]
            # print(list_of_left_x[:min_num_classes], list_of_right_x[:min_num_classes])

            for index, (left_x, right_x) in enumerate(zip(list_of_left_x[:min_num_classes], list_of_right_x[:min_num_classes])):
                if left_classes[0][index] == right_classes[0][index]:
                    pixel_dx = left_x - right_x
                    dx = pixel_dx * pitch * 1920
                    distance = (camera_distance * focal_length) / dx
                    if distance < 0:
                        # print(out_file_name)
                        print("left class:{} num of left {}\n".format(left_classes, nums_l[0]))
                        print("right class:{} num of right {}\n".format(right_classes, nums_r[0]))
                        print(distance)

            # # Uklanja dimenzije veličine 1, npr. (28, 28, 3, 1) ->  (28, 28, 3)
            # left_image = np.squeeze(left_image) 
            # out_img = draw_outputs(left_image, boxes, scores, classes, nums, class_names)

            # # Čuvanje rezultata u datoteku
            # out_file_name = './out/desna.png'
            # cv2.imwrite(out_file_name, out_img)

if __name__ == '__main__':
    main()