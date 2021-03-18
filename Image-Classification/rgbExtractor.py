# 25/02/21
# ML-Lab (4) | Extracting RGB data from image.  
# @author Dibyasom Puhan

# Import dependencies.
import cv2
import numpy as np
import os
import pandas
import pickle
from sklearn.model_selection import train_test_split


#Reading the images.

def extractFeatures(classes):
    extractedFeatures = {
        'R_avg_G_avg_B_avg': [],
        'RGB_avg': [],
        'label': []
    }
    for className in classes:
        try:
            path = './Resources/'+className+'/'
            if(os.path.exists(path)):
                covidDataset = os.scandir(path)
                for entry in covidDataset:
                    print("Reading img ...", entry.name)
                    img = cv2.imread(path+entry.name)

                    # Resclae the img, whilst mantaining the aspect-ratio.
                    print("Rescaling ...")
                    img = cv2.resize(img, (180,144))

                    # Extract RGB-avg and append to feature vector.
                    rgbAvgFeatureMat = np.array([
                        [rgbPixel[0] for pixelRow in img for rgbPixel in pixelRow], 
                        [rgbPixel[1] for pixelRow in img for rgbPixel in pixelRow], 
                        [rgbPixel[2] for pixelRow in img for rgbPixel in pixelRow]])
                    rgbAvgFeatureVect = np.array([
                        sum(rgbAvgFeatureMat[0])/len(rgbAvgFeatureMat[0]),
                        sum(rgbAvgFeatureMat[1])/len(rgbAvgFeatureMat[0]),
                        sum(rgbAvgFeatureMat[2])/len(rgbAvgFeatureMat[0])])

                    rgbFeaturMat = np.array([(int(rgbPixel[0])+int(rgbPixel[1])+int(rgbPixel[2]))//3  for pixelRow in img for rgbPixel in pixelRow]) #RGB channel avg.
                    extractedFeatures['RGB_avg'].append(rgbFeaturMat)
                    extractedFeatures['R_avg_G_avg_B_avg'].append(rgbAvgFeatureVect)
                    extractedFeatures['label'].append(className)
                
        except OSError:
            print("Image doesnot exist in specifiesd path, exiting ...")
            exit()
    return extractedFeatures

def saveFeatureMat(extractedFeatures):
    try: 
        featureFile = open('./ExtractedFeatures/features.pickle', 'wb') 
        pickle.dump(extractedFeatures, featureFile) 
        featureFile.close() 
    
    except pickle.PicklingError: 
        print("Serializing unsuccessful.")

def loadFeatureMat():
    try:
        featureFile = open("./ExtractedFeatures/features.pickle", "rb")
        featureMat = pickle.load(featureFile)
        return featureMat

    except pickle.UnpicklingError:
        print("Deserializing unsuccessful.")

# featureDict = extractFeatures(['COVID', 'NONCOVID'])
# saveFeatureMat(featureDict)
# print('Saved features into pickle file <3')
featureMat = loadFeatureMat()
# print("-"*30,"\nFeatures extracted successfully.")
print("Dataset size: {}\nFeatures: {}".format(str(len(featureMat['label'])), str(len(featureMat['RGB_avg'][0]))))
print(featureMat['R_avg_G_avg_B_avg'])

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(featureMat['RGB_avg'], featureMat['label'], test_size = 0.25, random_state = 0) # RGB_avg
# X_train, X_test, y_train, y_test = train_test_split(featureMat['R_avg_G_avg_B_avg'], featureMat['label'], test_size = 0.25, random_state = 0) # RGB_avg

# Resolve scaling issues.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
max_acc, best_k = 0, 0
for k in range(3, 10, 2):
    print("\nImplementing KNN, K = {}".format(str(k)))
    classifier = KNeighborsClassifier(n_neighbors = k)

    # K fold cross-validation ->
    k_fold_cross_validation = 10
    scores = cross_val_score(classifier, X_test, y_test, cv=k_fold_cross_validation)
    print(scores)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix.")

    acc = np.sum(scores)*100/k_fold_cross_validation
    if acc > max_acc:
        max_acc = acc
        best_k = k

    print("Accuracy: {}%\n".format(str(acc)), "-"*20)
    print(cm)
print("\nMax accuracy ({}%) for k = {}".format(str(max_acc), str(best_k)))
# #Display read image.
# cv2.imshow("Input Image", img)

# #Extract RGB patterns.

    
# cv2.waitKey(0) 
# cv2.destroyAllWindows()


