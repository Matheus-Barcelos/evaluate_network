import cv2
import os
import sys
import configparser
import dnn

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
        image = dataset[idx]
        
        filePathSplitted = os.path.splitext(image)
        annotations = []
        annotationFile = open(filePathSplitted[0]+".txt", 'r')
        
            
    def len(self):
        return len(self._train)+len(self._test)
    
    def get(self, idx):
        if(idx < len(self._train)):
            return self._train[idx]
        else:
            return self._test[idx-len(self._train)]
        
    def lenTest(self):
        return len(self._test)
    
    

if __name__=="__main__":
    dataset = DatasetDarknet("/media/data/Projetos/Madesa/wood_plate/obj.data")
    dataset._get_item(0, dataset._train)