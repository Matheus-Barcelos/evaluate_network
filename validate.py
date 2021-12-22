import dnn
import dataset
import os
import numpy
import cv2


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
    

def valid(model, data):
    tpP = fpP = 0
    for idx in [20]:
        image, annotations = data.get_test(idx)
        detections = model.inference(image)


        iouMat = numpy.zeros((len(annotations), len(detections)))

        for i in range(len(annotations)):
            for j in range(len(detections)):
                iouMat[i,j] = calcIOU(detections[j]["roi"], annotations[i]["roi"])
        
        print(iouMat)
        
        comp = 1-iouMat #convertendo problema de maximização em problema de minimização

        #for det in annotations:
        #    cv2.rectangle(image, det["roi"], (0,255,0))

        #for det in detections:
        #    cv2.rectangle(image, det["roi"], (0,0,255))
        #cv2.imwrite("det.png", image)
        
        











if __name__ == "__main__":
    basePath = "D:/Datasets/madesa"
    data = dataset.DatasetDarknet(os.path.join(basePath,"obj.data"))
    model = dnn.Yolo(os.path.join(basePath,"yolo-tome_esse_modelo_seu_pau_no_cu.cfg"),
                     os.path.join(basePath,"backup/yolo-tome_esse_modelo_seu_pau_no_cu_last.weights"), 
                     data.get_labels(), (512,288), 0.5, 0.3)
    
    valid(model, data)
    
