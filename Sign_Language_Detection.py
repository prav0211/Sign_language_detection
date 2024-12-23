import cv2
import numpy as np
from googletrans import Translator
from keras.models import load_model
import joblib
import cv2 as cv

# load cnn!
classifier = load_model('asl_cnn3.h5')
from keras.preprocessing import image

# final word
str_res = ""

# extract frames!
cap = cv.VideoCapture("C:\Users\prave\Documents\Mini project 1 sign")
clf = joblib.load('frame_svm.pkl')  # load the pickle file
i = 0
counter = 0
f_ct = 0
while (cap.isOpened()):
    # cv.waitKey(2000)
    target = 30  # extract every 30th frame!
    if counter == target:
        ret, frame = cap.read()
        if ret == False:
            break
        if i == 0:
            cv.imwrite('soop' + str(f_ct) + '.jpg', frame)
            frame = cv.rotate(frame, cv.cv2.ROTATE_90_CLOCKWISE)  # rotate
            cv.imwrite('hand' + str(f_ct) + '.jpg', frame)
            j = i
            i = i + 1
            f_ct = f_ct + 1  # unique images are saved as hand1,hand2,.. in order!

        if i > 0:

            a1 = cv.imread('soop' + str(j) + '.jpg')
            a2 = frame
            a1 = cv.normalize(a1, None, 0, 255, cv.NORM_MINMAX).astype('uint8')  # compression!
            a2 = cv.normalize(a2, None, 0, 255, cv.NORM_MINMAX).astype('uint8')  # and for easy img to array conversion!

            # load sift for comparing images!
            sift = cv.xfeatures2d.SIFT_create()

            # keypoints and coressponding descriptors in the 2 images-features
            kp_1, desc_1 = sift.detectAndCompute(a1, None)
            kp_2, desc_2 = sift.detectAndCompute(a2, None)

            # flann based comparison method betweeb the features of the 2 images:
            # brute force would be to compare all 1000 keypoints of 1 image to the other
            index_params = dict(algorithm=0, trees=5)
            search_params = dict()
            flann = cv.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(desc_1, desc_2, k=2)  # finding matches!
            # print(len(matches))
            # result = drawMatchesKnn(a1,kp_1,a2,kp_2,matches,None)
            # cv2.imshow(result)

            # we only need good points, m->first image keypoints, n->second
            good_points = []
            for m, n in matches:
                if m.distance < 0.6 * n.distance:  # lower the distance, better is the match!
                    good_points.append(m)

            # result = drawMatchesKnn(a1,kp_1,a2,kp_2,good_points,None) #comparitively reduced from matches

            number_keypoints = 0
            if len(kp_1) < len(kp_2):
                number_keypoints = len(kp_1)
            else:
                number_keypoints = len(kp_2)

            similarity = len(good_points) / number_keypoints

            op = clf.predict([[similarity]])

            # if svm output is 0, write as a new image!
            if (op == 0):
                cv.imwrite('soop' + str(i) + '.jpg', frame)
                frame = cv.rotate(frame, cv.cv2.ROTATE_90_CLOCKWISE)
                cv.imwrite('hand' + str(f_ct) + '.jpg', frame)
                j = i
                f_ct = f_ct + 1
            i = i + 1
            counter = 0

    # if not the 30th frame!
    else:
        ret = cap.grab()
        if ret == False:
            break
        counter += 1
    # j = i
    # i = i+1

cap.release()
cv.destroyAllWindows

# get the letter for the unique images!-hu moments!
i = 0
while i < f_ct:
    im = cv2.imread("hand" + str(i) + ".jpg")
    i = i + 1
    # skin segmentation!
    lower_skin = np.array([0, 48, 80])
    upper_skin = np.array([20, 255, 255])
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    save_img = cv2.resize(mask, (150, 150))
    cv2.imwrite('1.png', save_img)

    test_image = image.load_img('1.png', target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    # classifier using hu moments and other features like noise in between skin.
    if result[0][0] == 1:
        str_res += ('A')
    elif result[0][1] == 1:
        str_res += ('B')
    elif result[0][2] == 1:
        str_res += ('C')
    elif result[0][3] == 1:
        str_res += ('D')
    elif result[0][4] == 1:
        str_res += ('E')
    elif result[0][5] == 1:
        str_res += ('F')
    elif result[0][6] == 1:
        str_res += ('G')
    elif result[0][7] == 1:
        str_res += ('H')
    elif result[0][8] == 1:
        str_res += ('I')
    elif result[0][9] == 1:
        str_res += ('J')
    elif result[0][10] == 1:
        str_res += ('K')
    elif result[0][11] == 1:
        str_res += ('L')
    elif result[0][12] == 1:
        str_res += ('M')
    elif result[0][13] == 1:
        str_res += ('N')
    elif result[0][14] == 1:
        str_res += ('O')
    elif result[0][15] == 1:
        str_res += ('P')
    elif result[0][16] == 1:
        str_res += ('Q')
    elif result[0][17] == 1:
        str_res += ('R')
    elif result[0][18] == 1:
        str_res += ('S')
    elif result[0][19] == 1:
        str_res += ('T')
    elif result[0][20] == 1:
        str_res += ('U')
    elif result[0][21] == 1:
        str_res += ('V')
    elif result[0][22] == 1:
        str_res += ('W')
    elif result[0][23] == 1:
        str_res += ('X')
    elif result[0][24] == 1:
        str_res += ('Y')
    elif result[0][25] == 1:
        str_res += ('Z')

print("The word depicted is- " + str_res)

# translate the word into mother tongue!-using googletranslator!
while True:
    print("1. Tamil")
    print("2. Hindi")
    print("3. Telugu")
    print("4. Kannada")
    print("5. Malayalam")
    print("6. Exit")
    op = input("Please enter the option of the translated language-")
    lang = ""
    src = ""
    if op == "1":
        lang = 'ta'
        src = "Tamil"
    elif op == "2":
        lang = 'hi'
        src = "Hindi"
    elif op == "3":
        lang = 'te'
        src = "Telugu"
    elif op == "4":
        lang = 'kn'
        src = "Kannada"
    elif op == "5":
        lang = 'ml'
        src = "Malayalam"
    else:
        print("Exiting...")
        break
    translator = Translator()
    translator1 = Translator()
    sent = translator.translate(str_res.lower(), lang, 'en')
    # sent1 = translator1.translate(src,lang,'en')
    print(src + "- " + sent.text)
    # print(sent1.text+" - "+sent.text)
    print("Pronunciation- " + sent.pronunciation)
