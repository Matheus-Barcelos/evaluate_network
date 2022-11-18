import dnn
import dataset
import os
import numpy
from hungerian import hungarian_algorithm
import cv2
import tqdm
import sys


def calcIOU(box1, box2):
    left = max(box1[0], box2[0])
    right = min(box1[0] + box1[2], box2[0] + box2[2])

    upper = max(box1[1], box2[1])
    bottom = min(box1[1] + box1[3], box2[1] + box2[3])

    inter_w = right - left
    if inter_w < 0:
        inter_w = 0
    inter_h = bottom - upper
    if inter_h < 0:
        inter_h = 0

    inter_square = inter_w * inter_h
    union_square = (box1[2] * box1[3])+(box2[2] * box2[3])-inter_square


    return inter_square/union_square
    

def valid(model, data, threshold=0.6, write_images=False):
    tpP = fpP = fnP = 0

    if write_images:
        if not os.path.exists("false_negatives"):
            os.makedirs("false_negatives")
        if not os.path.exists("false_positives"):
            os.makedirs("false_positives")

    for idx in tqdm.tqdm(range(data.len_test())):
        image, annotations = data.get_test(idx)
        detections = model.inference(image)


        iouMat = numpy.zeros((len(annotations), len(detections)))

        for i in range(len(annotations)):
            for j in range(len(detections)):
                iouMat[i,j] = calcIOU(detections[j]["roi"], annotations[i]["roi"])
        
        comp = 1-iouMat #convertendo problema de maximização em problema de minimização
        matches = hungarian_algorithm(comp)
        filteredMatches = []
        for i,j in matches:
            if(iouMat[i,j] >= threshold and annotations[i]["label"] == detections[j]["label"]):
                filteredMatches.append((i,j))
        
        if write_images:
            for det in annotations:
                cv2.rectangle(image, det["roi"], (0,255,0))
            for det in detections:
                cv2.rectangle(image, det["roi"], (0,0,255))

            if(len(filteredMatches) != len(annotations)):
                cv2.imwrite("false_negatives/"+data.get_test_image_name(idx), image)
            if(len(filteredMatches) != len(detections)):
                cv2.imwrite("false_positives/"+data.get_test_image_name(idx), image)

        tpP += len(filteredMatches)
        fpP += len(detections) - len(filteredMatches)
        fnP += len(annotations) - len(filteredMatches)

    return tpP, fpP, fnP


def print_metrics(tpP, fpP, fnP):
    
    
    precision = tpP/(tpP+fpP)
    recall = tpP/(tpP+fnP)
    if precision+recall > 0:
        f1score = 2*((precision*recall)/(precision+recall))
    else:
        f1score = 0

    output = "TP: {} FP: {} FN: {}\nprecision: {}\nrecall: {}\nf1score: {}".format(tpP, fpP, fnP, precision, recall, f1score)
    print(output)
    file= open("results.txt",'w')
    file.write(output)
    file.close()
        
        
if __name__ == "__main__":
    basePath = sys.argv[1]
    data = dataset.DatasetDarknet(os.path.join(basePath,"obj.data"))
    model = dnn.Yolo(os.path.join(basePath,sys.argv[2]),
                     os.path.join(basePath,sys.argv[3]), 
                     data.get_labels(), (int(sys.argv[4]),int(sys.argv[5])), 0.5, 0.2)
    model.set_backend(cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA)
    
    tp, fp, fn = valid(model, data, 0.6, write_images=True)
    print_metrics(tp, fp, fn)
    
