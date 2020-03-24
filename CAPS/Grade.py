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
    file = open("textTA.txt", 'r', encoding="utf8")
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


pdfparser('grade.pdf')
res = read()
print(len(res))

pages = convert_from_path('grade.pdf', 500)

for page in pages:
    page.save('out.jpg', 'JPEG')


def resize(img, scale_per):
    width = int(img.shape[1] * scale_per / 100)
    height = int(img.shape[0] * scale_per / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def create_mask(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
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

    return vertical + horizal


def get_nghieng(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max = -1
    rect = []
    for cnt in contours:
        if cv2.contourArea(cnt) > max:
            max = cv2.contourArea(cnt)
            rect = cv2.minAreaRect(cnt)
    ng = rect[2]
    if ng < -45:
        ng = 90 + ng
    return ng


def fix_img(img, nghieng):
    cols = img.shape[1]
    rows = img.shape[0]
    m5 = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=nghieng, scale=1)
    return cv2.warpAffine(img, m5, (cols, rows))


def get_table(img, mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max = -1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > max:
            x_max, y_max, w_max, h_max = x, y, w, h
            max = cv2.contourArea(cnt)

    table = img[y_max:y_max + h_max, x_max:x_max + w_max]
    return table


def set_size(img, width, height):
    t = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    table = t[135:1995, :]
    return table


img = cv2.imread("out.jpg", 0)
img = resize(img, 50)
print(img.shape)
mask = create_mask(img)
#cv2.imshow('mask', mask)
ng = get_nghieng(mask)
new_img = fix_img(img, ng)
#cv2.imshow('new_img', new_img)
mask = create_mask(new_img)
#cv2.imshow('mask', mask)
table = get_table(new_img, mask)
print(table.shape)
img = set_size(table, 2200, 2000)
#cv2.imwrite('table.jpg', img)
k = img[:, 1645:1730]
cv2.imwrite("digit.jpg", k)

blur = cv2.GaussianBlur(img, (3, 3), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

cropped_thresh_img = []
cropped_origin_img = []
countours_img = []

NUM_ROWS = 19
START_ROW = 0
x1 = 1645 / 2200
x2 = 1730 / 2200
for i in range(START_ROW, NUM_ROWS):
    thresh1 = thresh[i * 62:(i + 1) * 62,
              round(x1 * (thresh.shape[1])):round(x2 * (thresh.shape[1]))]
    contours_thresh1, hierarchy_thresh1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    origin1 = img[round(i * 62):round((i + 1) * 62),
              round(x1 * (thresh.shape[1])):round(x2 * (thresh.shape[1]))]

    cropped_thresh_img.append(thresh1)
    cropped_origin_img.append(origin1)
    countours_img.append(contours_thresh1)
#    cv2.imshow("imgkk" + str(i), origin1)

print(len(countours_img))
res = []
for i, countour_img in enumerate(countours_img):
    for cnt in countour_img:
        if cv2.contourArea(cnt) > 40:
            x, y, w, h = cv2.boundingRect(cnt)
            if cropped_origin_img[i].shape[1] * 0.1 < x < cropped_origin_img[i].shape[1] * 0.9:
                answer = cropped_origin_img[i][y:y + h, x:x + w]
                answer = cv2.threshold(answer, 160, 255, cv2.THRESH_BINARY_INV)
                res.append(answer[1])

print(len(res))
for i in range(len(res)):
    cv2.imshow("imgkk" + str(i), res[i])

img = res[2]

frame = np.zeros((28, 28))


cv2.waitKey(0)
