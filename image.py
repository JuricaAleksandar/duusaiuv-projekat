# image.py - Primer detekcije objekata na jednoj slici koristeći YOLO v3 model
# (unapred istrenirani koeficijenti i konvertovani koeficijenti)

import os
import tensorflow as tf
from utils import load_class_names, output_boxes, draw_outputs, resize_image
import cv2
import numpy as np
from yolov3 import YOLOv3Net

model_size = (416, 416,3)   # Očekivani ulazni format za dati model i istrenirane koeficijente
num_classes = 80            # Broj klasa nad kojima je mreža trenirana  
class_name = './data/coco.names'    # Putanja do datoteke koja sadrži imena klasa
max_output_size = 40                # Najveći broj okvira koje želimo da dobijemo za sve klase ukupno
max_output_size_per_class= 20       # Najveći broj okvira koje želimo da dobijemo po klasi
iou_threshold = 0.6                # Prag mere preklapanja dva okvira
confidence_threshold = 0.55          # Prag pouzdanosti prisustva objekta 

cfgfile = './cfg/yolov3.cfg'                  # Putanja do YOLO v3 konfiguracione datoteke
weightfile = './weights/yolov3_weights.tf'    # Putanja do datoteke koja sadrži istrenirane koeficijente u TensorFlow formatu
left_img_path = "./data/image_L"
right_img_path = "./data/image_R"

#focal_length = 0.01206
focal_length = 0.00754
camera_distance = 0.54
pitch = 0.00000586

def main():

    # Kreiranje modela
    model = YOLOv3Net(cfgfile, model_size, num_classes)
    # Učitavanje istreniranih koeficijenata u model
    model.load_weights(weightfile)
    # Učitavanje imena klasa
    class_names = load_class_names(class_name)

    # Učitavanje ulazne slike i predobrada u format koji očekuje model
    left_img_names = os.listdir(left_img_path)
    right_img_names = os.listdir(right_img_path)

    left_image = cv2.imread(os.path.join(left_img_path, left_img_names[4]))
    left_image = np.array(left_image)
    left_image = tf.expand_dims(left_image, 0) # Dodavanje još jedne dimenzije, jer je ulaz definsian kao batch slika
    resized_left_image = resize_image(left_image, (model_size[0],model_size[1]))

    # Inferencija nad ulaznom slikom
    # izlazne predikcije pred - skup vektora (10647), gde svaki odgovara jednom okviru lokacije objekta 
    pred = model.predict(resized_left_image)

    # Određivanje okvira oko detektovanih objekata (za određene pragove)
    boxes, scores, classes, nums = output_boxes( \
        pred, model_size,
        max_output_size=max_output_size,
        max_output_size_per_class=max_output_size_per_class,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold)

    left_width = np.array(boxes[0])[0][2] - np.array(boxes[0])[0][0]
    left_x = np.array(boxes[0])[0][0] + left_width/2

    # Uklanja dimenzije veličine 1, npr. (28, 28, 3, 1) ->  (28, 28, 3)
    left_image = np.squeeze(left_image) 
    out_img = draw_outputs(left_image, boxes, scores, classes, nums, class_names)

    # Čuvanje rezultata u datoteku
    out_file_name = './out/leva.png'
    cv2.imwrite(out_file_name, out_img)

    ######### DESNA SLIKA ##############

    left_image = cv2.imread(os.path.join(right_img_path, right_img_names[4]))
    left_image = np.array(left_image)
    left_image = tf.expand_dims(left_image, 0) # Dodavanje još jedne dimenzije, jer je ulaz definsian kao batch slika
    resized_left_image = resize_image(left_image, (model_size[0],model_size[1]))

    # Inferencija nad ulaznom slikom
    # izlazne predikcije pred - skup vektora (10647), gde svaki odgovara jednom okviru lokacije objekta 
    pred = model.predict(resized_left_image)

    # Određivanje okvira oko detektovanih objekata (za određene pragove)
    boxes, scores, classes, nums = output_boxes( \
        pred, model_size,
        max_output_size=max_output_size,
        max_output_size_per_class=max_output_size_per_class,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold)

    right_width = np.array(boxes[0])[0][2] - np.array(boxes[0])[0][0]
    right_x = np.array(boxes[0])[0][0] + right_width/2
    pixel_dx = left_x - right_x
    dx = pixel_dx * pitch * 1762
    distance = (camera_distance * focal_length) / dx
    print(distance)

    # Uklanja dimenzije veličine 1, npr. (28, 28, 3, 1) ->  (28, 28, 3)
    left_image = np.squeeze(left_image) 
    out_img = draw_outputs(left_image, boxes, scores, classes, nums, class_names)

    # Čuvanje rezultata u datoteku
    out_file_name = './out/desna.png'
    cv2.imwrite(out_file_name, out_img)

    # Prikaz rezultata na ekran
    #cv2.imshow(out_file_name, out_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


if __name__ == '__main__':
    main()