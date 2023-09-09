from flask import Flask, request, Response, jsonify
from flask_cors import CORS

import torch
from torch import nn
from torchvision import transforms
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import open_clip
from sentence_transformers import util
from PIL import Image
from models.clipseg import CLIPDensePredT
import cv2
import urllib.request

app = Flask(__name__)
CORS(app)

# Load Model
print("Loading CLIPSeg...")
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
model.load_state_dict(torch.load('./model/rd64-uni-refined.pth', map_location=torch.device('cpu')), strict=False)
model.eval();

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Load Clip for Siamese
print("Loading Siamese...")
clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
clip.to(device)

print(device)

# Similarity Metrics
def imageEncoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = clip.encode_image(img1)
    return img1
def generateScore(image1, image2):
    img1 = imageEncoder(image1)
    img2 = imageEncoder(image2)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0])*100, 2)
    return score

# Common Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((352, 352)),
])

# Health check route
@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    print("/predict request")
    req_json = request.get_json()
    json_instances = req_json["instances"]

    query_List =[]
    target_List = []

    # Fetch Images from line
    for j in json_instances:
        resp = urllib.request.urlopen(j['query'])
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        query_List.append(cv2.imdecode(image, cv2.IMREAD_COLOR))

        resp = urllib.request.urlopen(j['target'])
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        target_List.append(cv2.imdecode(image, cv2.IMREAD_COLOR))

    response_list = []
    for query, target in zip(query_List, target_List):

        img = transform(query).unsqueeze(0)

        # Prepare Target
        target_clean = target.copy()
        _ , mask = cv2.threshold(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), 253, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.erode(mask, np.ones((3,3)))
        mask = cv2.dilate(mask, np.ones((3,3)), iterations=3)
        target = cv2.bitwise_and(target, target, mask=mask.astype(np.uint8))
        target_img = transform(target).unsqueeze(0)

        # Inference
        with torch.no_grad():
            preds = model(img, target_img)[0]

        # Process
        heatMap = torch.sigmoid(preds[0][0]).numpy()
        heatMap = (heatMap*255).astype(np.uint8)
        heatMap2 = heatMap.copy()
        _, heatMap = cv2.threshold(heatMap, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        raw = query  
        raw = raw[:, :, ::-1].copy() 

        contours, _ = cv2.findContours((heatMap*255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_cnt = None
        max_mean = 0
        max_score = 0
        #Scale cnts
        for cnt in contours:
            mask = np.zeros((352, 352, 1), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, (1), -1)
            # masked = np.ma.masked_array(heatMap, mask=mask)
            mean = cv2.meanStdDev(heatMap2, mask=mask)
            cnt[:,0,:] = cnt[:,0,:] * np.array([t/352 for t in raw.shape[:2][::-1]])
            x, y, w, h = cv2.boundingRect(cnt)
            if x > 0 and y > 0 and w > 0 and h > 0:
                crop = raw[y:y+h, x:x+w]

                diff = h - w
                if diff > 0:
                    left = int(np.ceil(diff/2))
                    right = int(np.floor(diff/2))
                    top = 0
                    bot = 0
                elif diff < 0:
                    top = int(np.ceil(-diff/2))
                    bot = int(np.floor(-diff/2))
                    left = 0
                    right = 0
                else:
                    left, right, top, bot = 0,0,0,0
                crop = cv2.copyMakeBorder(crop, top, bot, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
                crop = cv2.resize(crop, (512, 512))
                
                h, w = target_clean.shape[:2]
                diff = h - w
                if diff > 0:
                    left = int(np.ceil(diff/2))
                    right = int(np.floor(diff/2))
                    top = 0
                    bot = 0
                elif diff < 0:
                    top = int(np.ceil(-diff/2))
                    bot = int(np.floor(-diff/2))
                    left = 0
                    right = 0
                else:
                    left, right, top, bot = 0,0,0,0
                tgt = cv2.copyMakeBorder(target_clean, top, bot, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255)) 
                tgt = cv2.resize(tgt, (512, 512))
                
                if generateScore(crop, tgt) > 30:
                    if mean[0] > max_mean:
                        max_mean = mean[0]
                        max_cnt = cnt
                        max_score = generateScore(crop, tgt)

        if max_cnt is not None:
            if cv2.contourArea(max_cnt) > 2000:
                x, y, w, h = cv2.boundingRect(max_cnt)
                response_list.append([x,y,w,h])
                continue
        response_list.append(None)

    return jsonify({
        "predictions": response_list
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
