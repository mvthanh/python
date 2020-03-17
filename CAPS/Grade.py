from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdf2image import convert_from_path
import numpy as np
import io
import cv2


def pdfparser(data):
    fp = open(data, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data = retstr.getvalue()
    file = open("textTA.txt", "wb")
    file.write(data.encode())
    file.close()


def read():
    file = open("textTA.txt", 'r')
    result = []
    for line in file:
        line = line.strip()
        if len(line) == 0:
            continue
        else:
            l = line.split(' ')
            for it in l:
                if it.isdigit() and int(it) > 100000000:
                    print(len(result) + 1, it)
                    result.append(it)
    return result


'''
pdfparser('grade.pdf')
res = read()
print(len(res))
'''
pages = convert_from_path('grade.pdf', 500)

for page in pages:
    page.save('out.jpg', 'JPEG')

img = cv2.imread("out.jpg", 0)

scale_percent = 50
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
print(img.shape)

blur = cv2.GaussianBlur(img, (1, 1), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
# cv2.imshow('img', thresh)

horizal = thresh
vertical = thresh

scale_height = 30
scale_long = 20

long = int(img.shape[1] / scale_long)
height = int(img.shape[0] / scale_height)

horizalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (long, 1))
horizal = cv2.erode(horizal, horizalStructure, (-1, -1))
horizal = cv2.dilate(horizal, horizalStructure, (-1, -1))

verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height))
vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))

mask = vertical + horizal
# cv2.imshow('img1', mask)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

max = -1
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if cv2.contourArea(cnt) > max:
        x_max, y_max, w_max, h_max = x, y, w, h
        max = cv2.contourArea(cnt)
y_max += 100
h_max -= 100
table = img[y_max:y_max + h_max, x_max:x_max + w_max]
#cv2.imshow('img2', table)

cropped_thresh_img = []
cropped_origin_img = []
countours_img = []

NUM_ROWS = 34
START_ROW = 0
x1 = 470 / 723
x2 = 530 / 723
for i in range(START_ROW, NUM_ROWS):
    thresh1 = thresh[y_max + round(i * h_max / NUM_ROWS):y_max + round((i + 1) * h_max / NUM_ROWS),
              round(x1 * (thresh.shape[1])):round(x2 * (thresh.shape[1]))]
    contours_thresh1, hierarchy_thresh1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    origin1 = img[y_max + round(i * h_max / NUM_ROWS):y_max + round((i + 1) * h_max / NUM_ROWS),
              round(x1 * (thresh.shape[1])):round(x2 * (thresh.shape[1]))]

    cropped_thresh_img.append(thresh1)
    cropped_origin_img.append(origin1)
    countours_img.append(contours_thresh1)


print(len(countours_img))
res = []
for i, countour_img in enumerate(countours_img):
    for cnt in countour_img:
        if cv2.contourArea(cnt) > 30:
            x, y, w, h = cv2.boundingRect(cnt)
            if cropped_origin_img[i].shape[1] * 0.1 < x < cropped_origin_img[i].shape[1] * 0.9:
                answer = cropped_origin_img[i][y:y + h, x:x + w]
                answer = cv2.threshold(answer, 160, 255, cv2.THRESH_BINARY_INV)[1]
                res.append(answer)

print(len(res))
img = res[2]


frame = np.zeros((28, 28))

x = round((28-img.shape[0])/2)
y = round((28-img.shape[1])/2)
print(x, y)
frame[x: x + img.shape[0], y: y + img.shape[1]] = img

print(frame)
cv2.imshow("img", frame)
cv2.waitKey(0)
