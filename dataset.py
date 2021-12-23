import cv2
import os
import configparser

def readListFromFile(filePath):
    list = []
    file = open(filePath)
    for l in file:
        list.append(l.rstrip())
    return list

class DatasetDarknet:
    def __init__(self, pathToDataFile):
        
        config = configparser.ConfigParser()
        config.readfp(open(pathToDataFile))
        
        self._workDir = os.path.dirname(pathToDataFile)
        
        self._train = self._read_file_list(config.get('Project','train'))
        self._test = self._read_file_list(config.get('Project','valid'))
        
        labels_file = config.get('Project','names')
        if(not os.path.isabs(labels_file)):
            labels_file = os.path.join(self._workDir, labels_file)
        self._labels = readListFromFile(labels_file)
    
    def _read_file_list(self, filePath):
        
        if(not os.path.isabs(filePath)):
            train_file = os.path.join(self._workDir, filePath)
        fileList = readListFromFile(train_file)
        files = []
        for f in fileList:
            if(os.path.isabs(f)):
                files.append(f)
            else:
                files.append(os.path.join(self._workDir,f))
        return files
    
    def _get_item(self, idx, dataset):
        imagePath = dataset[idx]
        filePathSplitted = os.path.splitext(imagePath)
        image = cv2.imread(imagePath)
        annotations = []
        annotationFile = open(filePathSplitted[0]+".txt", 'r')
        for l in annotationFile:
            l = l.rstrip()
            info = l.split(" ")
            annotation = {}
            annotation["label"] = self._labels[int(info[0])]

            x = int(float(info[1])*image.shape[1])
            w = int(float(info[3])*image.shape[1])
            x = int(x - (w/2))

            y = int(float(info[2])*image.shape[0])
            h = int(float(info[4])*image.shape[0])
            y = int(y - (h/2))
            annotation["roi"] = [x,y,w,h]
            annotations.append(annotation)
        
        return image, annotations

    def len(self):
        return len(self._train)+len(self._test)
    
    def get(self, idx):
        if(idx < len(self._train)):
            return self._get_item(idx, self._train)
        else:
            return self._get_item(idx-len(self._train), self._test)

    def len_train(self):
        return len(self._train)
        
    def get_train(self, idx):
        return self._get_item(idx, self._train)
    
    def get_train_image_name(self, idx):
        return os.path.basename(self._train[idx])

    def len_test(self):
        return len(self._test)
        
    def get_test(self, idx):
        return self._get_item(idx, self._test)

    def get_test_image_name(self, idx):
        return os.path.basename(self._test[idx])


    def get_labels(self):
        return self._labels
    
    
if __name__=="__main__":
    basePath = "D:/Datasets/madesa"
    dataset = DatasetDarknet(os.path.join(basePath,"obj.data"))
    image, annotations = dataset.get_test(10)
    print(dataset.get_test_image_name(0))
    for annotation in annotations:
        cv2.rectangle(image, annotation["roi"], (0,255,0))
    cv2.imwrite("anot.png", image)