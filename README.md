# Sign_Language_Detection
This project is used to process a video depicting a word in sign language and gives the translation into one's regional language.

ASL_alphabet_recognition.py:
Train the model to recognise sign language alphabets using Convolutional Neural Network(CNN).

Sign_Language_Detection.py: 
Takes in the input video, Extracts the unique frames using Scale Invariant Feature Transform(SIFT) and Support Vector Machine(SVM).
Performs Image preprocessing using Skin Segmentation and predict the alphabet using the trained model.
The predicted alphabets are combined together and translated into one's regional language using Google Translate API.

Unique_frame_detection_TRAIN.py:
Train the model to classify unique frames.

Unique_frame_detection_TEST.py:
Test file to classify unique frames.

Metrics.py:
To check the accuracy of the svm model.




