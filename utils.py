# utils.py - skup korisnih funkcija

import tensorflow as tf
import numpy as np
import cv2

def resize_image(inputs, modelsize):
    #Skalira ulazne slike tako da odgovaraju veličini modela.

    #Parametri:
    #    inputs: originalne ulazne slike
    #    modelsize: očekivana veličina ulazne slike u model (širina, visina) 

    #Povratna vrednost:
    #    inputs: skalirane slike

    inputs= tf.image.resize(inputs, modelsize)
    return inputs


def load_class_names(file_name):
    #Učitavanje imena klasa iz datoteke

    #Parametri:
    #    file_name: Ime i putanja do datoteke u kojoj se nalazi lista klasa  

    #Povratna vrednost:
    #    class_names: lista sa imenima klasa

    with open(file_name, 'r') as f:
        class_names = f.read().splitlines()
    return class_names


def non_max_suppression(inputs, model_size, max_output_size,
                        max_output_size_per_class, iou_threshold,
                        confidence_threshold):
    # Non-maximum suppression procedura.

    #Parametri:
    #    inputs: Okviri
    #    model_size: veličina ulaza u model
    #    max_output_size: Najveći broj okvira koje želimo da dobijemo za sve klase ukupno
    #    max_output_size_per_class: Najveći broj okvira koje želimo da dobijemo po klasi
    #    iou_threshold:  Prag mere preklapanja dva okvira
    #    confidence_threshold: Prag pouzdanosti prisustva objekta 

    #Povratna vrednost:
    #    boxes: Okviri nakon Non-max suppression procedure
    #    scores: Tenzor koji sadrži verovatnoću prisustva objekta za date okvire
    #    classes: Tenzor koji sadrži klase za date okvire
    #    valid_detections: Tenzor koji sadrži broj validnih detekcija za date okvire. 
    #                      Samo prvi unosi u prethodne izlazne tenzore su validni. Ostatak do najvećeg broja je dopunjen nulama.

    bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
    bbox=bbox/model_size[0]

    scores = confs * class_probs
    boxes, scores, classes, valid_detections = \
        tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1,
                                   tf.shape(scores)[-1])),
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold
    )
    return boxes, scores, classes, valid_detections


def output_boxes(inputs, model_size, max_output_size, max_output_size_per_class,
                 iou_threshold, confidence_threshold):
    # Izdvajanje manjeg broja okvira od svih okvira koji su izlaz konvolucione obrade
    # Kao prvi kriterijum, odbacuju se svi okviri čija je ocena verovatnoće po ispod određenog praga, na primer 0.7.
    # Kao drugi korak, određuje se mera preklapanja dva okvira gde se njihov presek deli sa unijom površina 
    # (eng. intersection over union IoU), Slika 4.40.
    # Okvir sa najvećom vrednošću p se zadržava dok okviri sa kojima se on preklapa i mera IoU prelazi određeni prag se odbacuju.
    # Ova procedure je poznata kao eng. non-maximum supression.

    #Parametri:
    #    inputs: skup vektora (10647), gde svaki odgovara jednom okviru lokacije objekta
    #    model_size: veličina ulaza u model
    #    max_output_size: Najveći broj okvira koje želimo da dobijemo za sve klase ukupno
    #    max_output_size_per_class: Najveći broj okvira koje želimo da dobijemo po klasi
    #    iou_threshold:  Prag mere preklapanja dva okvira
    #    confidence_threshold: Prag pouzdanosti prisustva objekta 

    #Povratna vrednost:
    #    boxes_dicts: Rečnik koji sadrži okvire, verovatnoće, klase i broj validnih detekcija.

    # print(inputs.shape) # Dimenzije ulaznog skupa vektora
    center_x, center_y, width, height, confidence, classes = \
        tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    top_left_x = center_x - width / 2.0
    top_left_y = center_y - height / 2.0
    bottom_right_x = center_x + width / 2.0
    bottom_right_y = center_y + height / 2.0

    inputs = tf.concat([top_left_x, top_left_y, bottom_right_x,
                        bottom_right_y, confidence, classes], axis=-1)

    boxes_dicts = non_max_suppression(inputs, model_size, max_output_size,
                                      max_output_size_per_class, iou_threshold, confidence_threshold)

    return boxes_dicts

def draw_outputs(img, boxes, objectness, classes, nums, class_names):
    #Iscrtavanje detektovanih objekata na slici (pravougaonih, ime klase i verovatnoća)

    #Parametri:
    #    img: Slika
    #    boxes: Okviri za iscrtavanje
    #    objectness: Verovatnoća prisustva detektovanih objekta
    #    classes: Klase detektovanih objekata
    #    nums: Broj detektovanih objekata
    #    class_names: Lista sa imenima klasa

    #Povratna vrednost:
    #    img: Izlazna slika

    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    boxes=np.array(boxes)

    for i in range(nums):
        x1y1 = tuple((boxes[i,0:2] * [img.shape[1],img.shape[0]]).astype(np.int32))
        x2y2 = tuple((boxes[i,2:4] * [img.shape[1],img.shape[0]]).astype(np.int32))

        img = cv2.rectangle(img, (x1y1), (x2y2), (255,0,0), 2)

        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
                          (x1y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    return img