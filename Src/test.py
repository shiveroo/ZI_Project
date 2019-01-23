import os
import tensorflow as tf
import numpy as np
import cv2

#sciezka do modelu i obrazow testowych
RETRAINED_LABELS_TXT_FILE_LOC = os.getcwd() + "/" + "retrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOC = os.getcwd() + "/" + "retrained_graph.pb"

TEST_IMAGES_DIR = os.getcwd() + "/test_images"

def main():

    if not checkIfNecessaryPathsAndFilesExist():
        return

    # pobranie listy klasyfikacji z pliku
    classifications = []
    for currentLine in tf.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
        classification = currentLine.rstrip()
        classifications.append(classification)
    print("classifications = " + str(classifications))

    # zaladowanie grafu
    with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
        # instancja obiektu grafu
        graphDef = tf.GraphDef()
        # Wczytanie "wytrenowanego grafu" do obiektu
        graphDef.ParseFromString(retrainedGraphFile.read())
        #imporotowanie grafu jako domyslny
        _ = tf.import_graph_def(graphDef, name='')

    # error jesli sciezka do folderu z obrazami nie prawidlowa
    if not os.path.isdir(TEST_IMAGES_DIR):
        print("bad directory")
        return

    with tf.Session() as sess:
        for fileName in os.listdir(TEST_IMAGES_DIR):
            if not (fileName.lower().endswith(".jpg") or fileName.lower().endswith(".jpeg")):
                continue

            print(fileName)

            #otwarcie pliku prze opencv
            imageFileWithPath = os.path.join(TEST_IMAGES_DIR, fileName)
            openCVImage = cv2.imread(imageFileWithPath)

            if openCVImage is None:
                print("unable to open " + fileName + " as an OpenCV image")
                continue

            finalTensor = sess.graph.get_tensor_by_name('final_result:0')

            # konwersja do obrazka zgodnego z tensorflow
            tfImage = np.array(openCVImage)[:, :, 0:3]
            
            # wykonanie predykcji
            predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

            # sortowanie od najbardziej zgodnych
            sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

            onMostLikelyPrediction = True
            #wykonanie dla kazdej predykcji
            for prediction in sortedPredictions:
                strClassification = classifications[prediction]

                confidence = predictions[0][prediction] #dokladnosc predykcji zaokraglona do dwoch miejsc

                if onMostLikelyPrediction:
                    scoreAsAPercent = confidence * 100.0
                    # przypisanie obiektu do kategori wraz z procentem pewnosci, pokazanie obrazu
                    print("it appears to be a flag of \n" + strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
                    writeResultOnImage(openCVImage, strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
                    cv2.imshow(fileName, openCVImage)
                    onMostLikelyPrediction = False

                #pewnosc predykcji
                print(strClassification + " (" +  "{0:.5f}".format(confidence) + ")")

            cv2.waitKey()
            cv2.destroyAllWindows()

    # zapisanie grafu do pliku
    tfFileWriter = tf.summary.FileWriter(os.getcwd())
    tfFileWriter.add_graph(sess.graph)
    tfFileWriter.close()

def checkIfNecessaryPathsAndFilesExist():
    if not os.path.exists(TEST_IMAGES_DIR):
        print('')
        print('ERROR: TEST_IMAGES_DIR "' + TEST_IMAGES_DIR + '" does not seem to exist')
        print('Did you set up the test images?')
        print('')
        return False

    if not os.path.exists(RETRAINED_LABELS_TXT_FILE_LOC):
        print('ERROR: RETRAINED_LABELS_TXT_FILE_LOC "' + RETRAINED_LABELS_TXT_FILE_LOC + '" does not seem to exist')
        return False

    if not os.path.exists(RETRAINED_GRAPH_PB_FILE_LOC):
        print('ERROR: RETRAINED_GRAPH_PB_FILE_LOC "' + RETRAINED_GRAPH_PB_FILE_LOC + '" does not seem to exist')
        return False

    return True

#wypisanie danych na zdjeciu, okreslenie czecionki
def writeResultOnImage(openCVImage, resultText):

    imageHeight, imageWidth, sceneNumChannels = openCVImage.shape

    fontFace = cv2.FONT_HERSHEY_TRIPLEX

    fontScale = 1.0
    fontThickness = 2

    fontThickness = int(fontThickness) # ??? czcionka musi byc "integer", wywala inaczej opencv

    upperLeftTextOriginX = int(imageWidth * 0.05)
    upperLeftTextOriginY = int(imageHeight * 0.05)

    textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
    textSizeWidth, textSizeHeight = textSize

    lowerLeftTextOriginX = upperLeftTextOriginX
    lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight
    COLOUR = (255.0, 0.0, 0.0)

    cv2.putText(openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, COLOUR, fontThickness)

if __name__ == "__main__":
    main()
