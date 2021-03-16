import torch
from pathlib import Path
import sys
import cv2
sys.path.append("..")
from models.model import get_tsn_model
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description='running inference on video')
parser.add_argument("weights", type=Path, help="weights file for model")
parser.add_argument("video_file", type=Path, help="path to video file")
parser.add_argument("json_file", type=Path, help="json file containing index to class mappings")

args = parser.parse_args()
weights = args.weights
video_file = args.video_file
json_file = args.json_file


def pre_process_img(img):
    img = cv2.resize(img,(tsn.input_size, tsn.input_size), interpolation=cv2.INTER_LINEAR)
    #convert to RGB..
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def get_class_name(out):
    #load json file that contains index to class name mapping ..

    with open(json_file, "r") as f:
        content = json.load(f)
 
    _, pred = out.topk(1, dim=-1, largest=True, sorted =True) #returns index of largest element
    pred = pred.item()

    class_name = [k  for k, v in content.items() if v == pred][0]
    return class_name



def infer(img_stack):
    img_tensor  = torch.from_numpy(img_stack)

    #normalize and permute
    img_tensor = (img_tensor.float()/255.0 - tsn.input_mean[0])/tsn.input_std[0]
    img_tensor = img_tensor.permute(2,0, 1)

    #add batch dimenstion
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        #run inference on img
        out, _ = tsn(img_tensor)
        class_name = get_class_name(out)
    return class_name


#load model and weights ..
tsn = get_tsn_model(base_model="resnet50", segment_count=8, tune_model=True)
tsn.eval()
w_dict = torch.load(weights)
tsn.load_state_dict(w_dict)

cap = cv2.VideoCapture(str(args.video_file))

#write video 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 10.0, (1280,720))

img_stack = []
num_segments = 8
while (cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break
    img_stack.append(frame.copy())
    if len(img_stack) == num_segments:
        images = list(map(pre_process_img,img_stack)) 
        images = np.stack(images, axis=2)
        images = images.reshape((images.shape[0], images.shape[1], -1))
        class_name = infer(images)  
        img_stack = []

        cv2.putText(frame, class_name, org= (frame.shape[1] -250, 55),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.5,
                    color=(255, 0, 0))
    out.write(frame)

    cv2.imshow("frame", frame)
    if cv2.waitKey(100) & 0xFF == ord('q'): #output at 10FPS.
        break

cap.release()
cv2.destroyAllWindows()
out.release()
