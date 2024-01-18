import cv2
import re
import json
import os
import torch
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import functools
import numpy as np
import face_detection
import io, tokenize
from augly.utils.base_paths import EMOJI_DIR
import augly.image as imaugs
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from transformers import (ViltProcessor, ViltForQuestionAnswering,
    OwlViTProcessor, OwlViTForObjectDetection,
    MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation,
    CLIPProcessor, CLIPModel, AutoProcessor, BlipForQuestionAnswering)
from diffusers import StableDiffusionInpaintPipeline
import torchvision
from torchvision import transforms

from .nms import nms
from visprog.vis_utils import html_embed_image, html_embed_video, html_colored_span, vis_masks
from tqdm import tqdm
from collections import defaultdict
import duckdb
import yaml
import pandas as pd
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils.visualize import plot_tracking, vis
from argparse import Namespace

config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))

def parse_step(step_str, partial=False):
    """
    partial = True: parse only output_var and step_name (i.e., module name)
    partial = False: additionally parse args
    """
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    output_var = tokens[0].string
    step_name = tokens[2].string
    parsed_result = dict(
        output_var=output_var,
        step_name=step_name)
    if partial:
        return parsed_result

    # Modified to allow arguments to be lists
    # arg_tokens = [token for token in tokens[4:-3] if token.string not in [',','=']]
    inside_arg_value = False
    arg_tokens = []
    arg_string = ''
    for token in tokens[4:-3]:
        if token.string == '[':
            arg_string += token.string
            inside_arg_value = True
        elif token.string == ']':
            arg_string += token.string
            arg_tokens.append(arg_string)
            arg_string = ''
            inside_arg_value = False
        elif token.string in [',', '=']:
            if inside_arg_value:
                arg_string += token.string
            else:
                continue
        else:
            if inside_arg_value:
                arg_string += token.string
            else:
                arg_tokens.append(token.string)

    num_tokens = len(arg_tokens) // 2
    args = dict()
    for i in range(num_tokens):
        args[arg_tokens[2*i]] = arg_tokens[2*i+1]
    parsed_result['args'] = args
    return parsed_result


def html_step_name(content):
    step_name = html_colored_span(content, 'red')
    return f'<b>{step_name}</b>'


def html_output(content):
    output = html_colored_span(content, 'green')
    return f'<b>{output}</b>'

def html_output_as_table(content):
    if isinstance(content,dict):
        key_row_str = '<th>key</th>'
        val_row_str = '<th>val</th>'
        for key, val in content.items():
            key_row_str += f'<th>{key}</th>'
            val_row_str += f'<th>{val}</th>'
        content = f'<table style="display:inline-table;"><tr>{key_row_str}</tr><tr>{val_row_str}</tr></table>'
        output = html_colored_span(content, 'green')
        return f'<b>{output}</b>'
    elif isinstance(content,list):
        row_str_dict = {}
        for col_idx, item in enumerate(content):
            if col_idx == 0:
                row_str_dict[''] = f'<th></th><th>{col_idx}</th>'
                for key, val in item.items():
                    row_str_dict[key] = f'<th>{key}</th><th>{val}</th>'
            else:
                row_str_dict[''] += f'<th>{col_idx}</th>'
                for key, val in item.items():
                    row_str_dict[key] += f'<th>{val}</th>'
        content = f'<table style="display:inline-table;">'
        for key, row_str in row_str_dict.items():
            content += f'<tr>{row_str}</tr>'
        content += '</table>'
        output = html_colored_span(content, 'green')
        return f'<b>{output}</b>'

def html_var_name(content):
    var_name = html_colored_span(content, 'blue')
    return f'<b>{var_name}</b>'


def html_arg_name(content):
    arg_name = html_colored_span(content, 'darkorange')
    return f'<b>{arg_name}</b>'


class EvalInterpreter():
    step_name = 'EVAL'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        step_input = eval(parse_result['args']['expr'])
        assert(step_name==self.step_name)
        return step_input, output_var

    def html(self,eval_expression,step_input,step_output,output_var):
        eval_expression = eval_expression.replace('{','').replace('}','')
        step_name = html_step_name(self.step_name)
        var_name = html_var_name(output_var)
        output = html_output(step_output)
        expr = html_arg_name('expression')
        return f"""<div>{var_name}={step_name}({expr}="{eval_expression}")={step_name}({expr}="{step_input}")={output}</div>"""

    def execute(self,prog_step,inspect=False):
        step_input, output_var = self.parse(prog_step)
        prog_state = dict()
        for var_name,var_value in prog_step.state.items():
            if isinstance(var_value,str):
                if var_value in ['yes','no']:
                    prog_state[var_name] = var_value=='yes'
                elif var_value.isdecimal():
                    prog_state[var_name] = var_value
                else:
                    prog_state[var_name] = f"'{var_value}'"
            elif isinstance(var_value, pd.DataFrame):
                prog_state[var_name] = var_value.values.tolist()
            else:
                prog_state[var_name] = var_value

        eval_expression = step_input

        if 'xor' in step_input:
            step_input = step_input.replace('xor','!=')

        step_input = step_input.format(**prog_state)
        step_output = eval(step_input)
        prog_step.state[output_var] = step_output
        if inspect:
            html_str = self.html(eval_expression, step_input, step_output, output_var)
            return step_output, html_str

        return step_output


class ResultInterpreter():
    step_name = 'RESULT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['args']['var']
        assert(step_name==self.step_name)
        return output_var

    def html(self,output,output_var):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        if isinstance(output, Image.Image):
            output = html_embed_image(output,300)
        else:
            output = html_output(output)

        return f"""<div>{step_name} -> {output_var} -> {output}</div>"""

    def execute(self,prog_step,inspect=False):
        output_var = self.parse(prog_step)
        output = prog_step.state[output_var]
        if inspect:
            html_str = self.html(output,output_var)
            return output, html_str

        return output


class VQAInterpreter():
    step_name = 'VQA'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-capfilt-large").to(self.device)
        self.model.eval()

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        args = parse_result['args']
        img_var = args['image']
        question = eval(args['question'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,question,output_var

    def predict(self,img,question):
        encoding = self.processor(img,question,return_tensors='pt')
        encoding = {k:v.to(self.device) for k,v in encoding.items()}
        with torch.no_grad():
            outputs = self.model.generate(**encoding)

        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def html(self,img,question,answer,output_var):
        step_name = html_step_name(self.step_name)
        img_str = html_embed_image(img)
        answer = html_output(answer)
        output_var = html_var_name(output_var)
        image_arg = html_arg_name('image')
        question_arg = html_arg_name('question')
        return f"""<div>{output_var}={step_name}({image_arg}={img_str},&nbsp;{question_arg}='{question}')={answer}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,question,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        answer = self.predict(img,question)
        prog_step.state[output_var] = answer
        if inspect:
            html_str = self.html(img, question, answer, output_var)
            return answer, html_str

        return answer


class LocInterpreter():
    step_name = 'LOC'

    def __init__(self,thresh=0.1,nms_thresh=0.5):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = OwlViTProcessor.from_pretrained(
            "google/owlvit-large-patch14")
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-large-patch14").to(self.device)
        self.model.eval()
        self.thresh = thresh
        self.nms_thresh = nms_thresh

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_name = eval(parse_result['args']['object'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_name,output_var

    def normalize_coord(self,bbox,img_size):
        w,h = img_size
        x1,y1,x2,y2 = [int(v) for v in bbox]
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(x2,w-1)
        y2 = min(y2,h-1)
        return [x1,y1,x2,y2]

    def predict(self,img,obj_name):
        encoding = self.processor(
            text=[[f'a photo of {obj_name}']],
            images=img,
            return_tensors='pt')
        encoding = {k:v.to(self.device) for k,v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            for k,v in outputs.items():
                if v is not None:
                    outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v

        target_sizes = torch.Tensor([img.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs,threshold=self.thresh,target_sizes=target_sizes)
        boxes, scores = results[0]["boxes"], results[0]["scores"]
        boxes = boxes.cpu().detach().numpy().tolist()
        scores = scores.cpu().detach().numpy().tolist()
        if len(boxes)==0:
            return []

        boxes, scores = zip(*sorted(zip(boxes,scores),key=lambda x: x[1],reverse=True))
        selected_boxes = []
        selected_scores = []
        for i in range(len(scores)):
            if scores[i] > self.thresh:
                coord = self.normalize_coord(boxes[i],img.size)
                selected_boxes.append(coord)
                selected_scores.append(scores[i])

        selected_boxes, selected_scores = nms(
            selected_boxes,selected_scores,self.nms_thresh)
        return selected_boxes

    def top_box(self,img):
        w,h = img.size
        return [0,0,w-1,int(h/2)]

    def bottom_box(self,img):
        w,h = img.size
        return [0,int(h/2),w-1,h-1]

    def left_box(self,img):
        w,h = img.size
        return [0,0,int(w/2),h-1]

    def right_box(self,img):
        w,h = img.size
        return [int(w/2),0,w-1,h-1]

    def box_image(self,img,boxes,highlight_best=True):
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        for i,box in enumerate(boxes):
            if i==0 and highlight_best:
                color = 'red'
            else:
                color = 'blue'

            draw.rectangle(box,outline=color,width=5)

        return img1

    def html(self,img,box_img,output_var,obj_name):
        step_name=html_step_name(self.step_name)
        obj_arg=html_arg_name('object')
        img_arg=html_arg_name('image')
        output_var=html_var_name(output_var)
        img=html_embed_image(img)
        box_img=html_embed_image(box_img,300)
        return f"<div>{output_var}={step_name}({img_arg}={img}, {obj_arg}='{obj_name}')={box_img}</div>"


    def extract_words(self, input_string):
        # This regular expression pattern matches any sequence of letters
        word_pattern = re.compile(r'\b\w+\b')

        # Find all substrings where the pattern matches and return them
        words = word_pattern.findall(input_string)

        return words

    def execute(self,prog_step,inspect=False):
        """
        Return bounding box
        """
        img_var,obj_name,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        if obj_name=='TOP':
            bboxes = [self.top_box(img)]
        elif obj_name=='BOTTOM':
            bboxes = [self.bottom_box(img)]
        elif obj_name=='LEFT':
            bboxes = [self.left_box(img)]
        elif obj_name=='RIGHT':
            bboxes = [self.right_box(img)]
        else:
            bboxes = self.predict(img,obj_name)

        box_img = self.box_image(img, bboxes)
        prog_step.state[output_var] = bboxes
        prog_step.state[output_var+'_IMAGE'] = box_img
        if inspect:
            html_str = self.html(img, box_img, output_var, obj_name)
            return bboxes, html_str

        return bboxes


class Loc2Interpreter(LocInterpreter):
    def execute(self,prog_step,inspect=False):
        """
        Return bounding box and object name
        """
        img_var,obj_name,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        bboxes = self.predict(img,obj_name)

        objs = []
        for box in bboxes:
            objs.append(dict(
                box=box,
                category=obj_name
            ))
        prog_step.state[output_var] = objs

        if inspect:
            box_img = self.box_image(img, bboxes, highlight_best=False)
            html_str = self.html(img, box_img, output_var, obj_name)
            return bboxes, html_str

        return objs


class CountInterpreter():
    step_name = 'COUNT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        box_var = parse_result['args']['box']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return box_var,output_var

    def html(self,box_img,output_var,count):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        box_arg = html_arg_name('bbox')
        box_img = html_embed_image(box_img)
        output = html_output(count)
        return f"""<div>{output_var}={step_name}({box_arg}={box_img})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        box_var,output_var = self.parse(prog_step)
        boxes = prog_step.state[box_var]
        count = len(boxes)
        prog_step.state[output_var] = count
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(box_img, output_var, count)
            return count, html_str

        return count


class CropInterpreter():
    step_name = 'CROP'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def expand_box(self,box,img_size,factor=1.5):
        W,H = img_size
        x1,y1,x2,y2 = box
        dw = int(factor*(x2-x1)/2)
        dh = int(factor*(y2-y1)/2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x1 = max(0,cx - dw)
        x2 = min(cx + dw,W)
        y1 = max(0,cy - dh)
        y2 = min(cy + dh,H)
        return [x1,y1,x2,y2]

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        box_var = parse_result['args']['box']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,box_var,output_var

    def html(self,img,out_img,output_var,box_img):
        img = html_embed_image(img)
        out_img = html_embed_image(out_img,300)
        box_img = html_embed_image(box_img)
        output_var = html_var_name(output_var)
        step_name = html_step_name(self.step_name)
        box_arg = html_arg_name('bbox')
        return f"""<div>{output_var}={step_name}({box_arg}={box_img})={out_img}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            box = self.expand_box(box, img.size)
            out_img = img.crop(box)
        else:
            box = []
            out_img = img

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img


class CropRightOfInterpreter(CropInterpreter):
    step_name = 'CROP_RIGHTOF'

    def right_of(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cx = int((x1+x2)/2)
        return [cx,0,w-1,h-1]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            right_box = self.right_of(box, img.size)
        else:
            w,h = img.size
            box = []
            right_box = [int(w/2),0,w-1,h-1]

        out_img = img.crop(right_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img


class CropLeftOfInterpreter(CropInterpreter):
    step_name = 'CROP_LEFTOF'

    def left_of(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cx = int((x1+x2)/2)
        return [0,0,cx,h-1]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            left_box = self.left_of(box, img.size)
        else:
            w,h = img.size
            box = []
            left_box = [0,0,int(w/2),h-1]

        out_img = img.crop(left_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img


class CropAboveInterpreter(CropInterpreter):
    step_name = 'CROP_ABOVE'

    def above(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cy = int((y1+y2)/2)
        return [0,0,w-1,cy]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            above_box = self.above(box, img.size)
        else:
            w,h = img.size
            box = []
            above_box = [0,0,int(w/2),h-1]

        out_img = img.crop(above_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img

class CropBelowInterpreter(CropInterpreter):
    step_name = 'CROP_BELOW'

    def below(self,box,img_size):
        w,h = img_size
        x1,y1,x2,y2 = box
        cy = int((y1+y2)/2)
        return [0,cy,w-1,h-1]

    def execute(self,prog_step,inspect=False):
        img_var,box_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        boxes = prog_step.state[box_var]
        if len(boxes) > 0:
            box = boxes[0]
            below_box = self.below(box, img.size)
        else:
            w,h = img.size
            box = []
            below_box = [0,0,int(w/2),h-1]

        out_img = img.crop(below_box)

        prog_step.state[output_var] = out_img
        if inspect:
            box_img = prog_step.state[box_var+'_IMAGE']
            html_str = self.html(img, out_img, output_var, box_img)
            return out_img, html_str

        return out_img

class CropFrontOfInterpreter(CropInterpreter):
    step_name = 'CROP_FRONTOF'

class CropInFrontInterpreter(CropInterpreter):
    step_name = 'CROP_INFRONT'

class CropInFrontOfInterpreter(CropInterpreter):
    step_name = 'CROP_INFRONTOF'

class CropBehindInterpreter(CropInterpreter):
    step_name = 'CROP_BEHIND'


class CropAheadInterpreter(CropInterpreter):
    step_name = 'CROP_AHEAD'


class SegmentInterpreter():
    step_name = 'SEG'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = MaskFormerFeatureExtractor.from_pretrained(
            "facebook/maskformer-swin-base-coco")
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(
            "facebook/maskformer-swin-base-coco").to(self.device)
        self.model.eval()

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,output_var

    def pred_seg(self,img):
        inputs = self.feature_extractor(images=img, return_tensors="pt")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        outputs = self.feature_extractor.post_process_panoptic_segmentation(outputs)[0]
        instance_map = outputs['segmentation'].cpu().numpy()
        objs = []
        print(outputs.keys())
        for seg in outputs['segments_info']:
            inst_id = seg['id']
            label_id = seg['label_id']
            category = self.model.config.id2label[label_id]
            mask = (instance_map==inst_id).astype(float)
            resized_mask = np.array(
                Image.fromarray(mask).resize(
                    img.size,resample=Image.BILINEAR))
            Y,X = np.where(resized_mask>0.5)
            x1,x2 = np.min(X), np.max(X)
            y1,y2 = np.min(Y), np.max(Y)
            num_pixels = np.sum(mask)
            objs.append(dict(
                mask=resized_mask,
                category=category,
                box=[x1,y1,x2,y2],
                inst_id=inst_id
            ))

        return objs

    def html(self,img_var,output_var,output):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        img_var = html_var_name(img_var)
        img_arg = html_arg_name('image')
        output = html_embed_image(output,300)
        return f"""<div>{output_var}={step_name}({img_arg}={img_var})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = self.pred_seg(img)
        prog_step.state[output_var] = objs
        if inspect:
            labels = [str(obj['inst_id'])+':'+obj['category'] for obj in objs]
            obj_img = vis_masks(img, objs, labels)
            html_str = self.html(img_var, output_var, obj_img)
            return objs, html_str

        return objs


class SelectInterpreter():
    step_name = 'SELECT'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14").to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        query = eval(parse_result['args']['query']).split(',')
        category = eval(parse_result['args']['category'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,query,category,output_var

    def calculate_sim(self,inputs):
        img_feats = self.model.get_image_features(inputs['pixel_values'])
        text_feats = self.model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats,text_feats.t())

    def query_obj(self,query,objs,img):
        images = [img.crop(obj['box']) for obj in objs]
        text = [f'a photo of {q}' for q in query]
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            scores = self.calculate_sim(inputs).cpu().numpy()

        obj_ids = scores.argmax(0)
        return [objs[i] for i in obj_ids]

    def html(self,img_var,obj_var,query,category,output_var,output):
        step_name = html_step_name(self.step_name)
        image_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        query_arg = html_arg_name('query')
        category_arg = html_arg_name('category')
        image_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        output = html_embed_image(output,300)
        return f"""<div>{output_var}={step_name}({image_arg}={image_var},{obj_arg}={obj_var},{query_arg}={query},{category_arg}={category})={output}</div>"""

    def query_string_match(self,objs,q):
        obj_cats = [obj['category'] for obj in objs]
        q = q.lower()
        for cat in [q,f'{q}-merged',f'{q}-other-merged']:
            if cat in obj_cats:
                return [obj for obj in objs if obj['category']==cat]

        return None

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,query,category,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        select_objs = []

        if category is not None:
            cat_objs = [obj for obj in objs if obj['category'] in category]
            if len(cat_objs) > 0:
                objs = cat_objs


        if category is None:
            for q in query:
                matches = self.query_string_match(objs, q)
                if matches is None:
                    continue

                select_objs += matches

        if query is not None and len(select_objs)==0:
            select_objs = self.query_obj(query, objs, img)

        prog_step.state[output_var] = select_objs
        if inspect:
            select_obj_img = vis_masks(img, select_objs)
            html_str = self.html(img_var, obj_var, query, category, output_var, select_obj_img)
            return select_objs, html_str

        return select_objs


class ColorpopInterpreter():
    step_name = 'COLORPOP'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,output_var

    def refine_mask(self,img,mask):
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        mask,_,_ = cv2.grabCut(
            img.astype(np.uint8),
            mask.astype(np.uint8),
            None,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_MASK)
        return mask.astype(float)

    def html(self,img_var,obj_var,output_var,output):
        step_name = html_step_name(self.step_name)
        img_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        img_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        output = html_embed_image(output,300)
        return f"""{output_var}={step_name}({img_arg}={img_var},{obj_arg}={obj_var})={output}"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        gimg = img.copy()
        gimg = gimg.convert('L').convert('RGB')
        gimg = np.array(gimg).astype(float)
        img = np.array(img).astype(float)
        for obj in objs:
            refined_mask = self.refine_mask(img, obj['mask'])
            mask = np.tile(refined_mask[:,:,np.newaxis],(1,1,3))
            gimg = mask*img + (1-mask)*gimg

        gimg = np.array(gimg).astype(np.uint8)
        gimg = Image.fromarray(gimg)
        prog_step.state[output_var] = gimg
        if inspect:
            html_str = self.html(img_var, obj_var, output_var, gimg)
            return gimg, html_str

        return gimg


class BgBlurInterpreter():
    step_name = 'BGBLUR'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,output_var

    def refine_mask(self,img,mask):
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        mask,_,_ = cv2.grabCut(
            img.astype(np.uint8),
            mask.astype(np.uint8),
            None,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_MASK)
        return mask.astype(float)

    def smoothen_mask(self,mask):
        mask = Image.fromarray(255*mask.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(radius = 5))
        return np.array(mask).astype(float)/255

    def html(self,img_var,obj_var,output_var,output):
        step_name = html_step_name(self.step_name)
        img_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        img_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        output = html_embed_image(output,300)
        return f"""{output_var}={step_name}({img_arg}={img_var},{obj_arg}={obj_var})={output}"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        bgimg = img.copy()
        bgimg = bgimg.filter(ImageFilter.GaussianBlur(radius = 2))
        bgimg = np.array(bgimg).astype(float)
        img = np.array(img).astype(float)
        for obj in objs:
            refined_mask = self.refine_mask(img, obj['mask'])
            mask = np.tile(refined_mask[:,:,np.newaxis],(1,1,3))
            mask = self.smoothen_mask(mask)
            bgimg = mask*img + (1-mask)*bgimg

        bgimg = np.array(bgimg).astype(np.uint8)
        bgimg = Image.fromarray(bgimg)
        prog_step.state[output_var] = bgimg
        if inspect:
            html_str = self.html(img_var, obj_var, output_var, bgimg)
            return bgimg, html_str

        return bgimg


class FaceDetInterpreter():
    step_name = 'FACEDET'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.model = face_detection.build_detector(
            "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,output_var

    def box_image(self,img,boxes):
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        for i,box in enumerate(boxes):
            draw.rectangle(box,outline='blue',width=5)

        return img1

    def enlarge_face(self,box,W,H,f=1.5):
        x1,y1,x2,y2 = box
        w = int((f-1)*(x2-x1)/2)
        h = int((f-1)*(y2-y1)/2)
        x1 = max(0,x1-w)
        y1 = max(0,y1-h)
        x2 = min(W,x2+w)
        y2 = min(H,y2+h)
        return [x1,y1,x2,y2]

    def det_face(self,img):
        with torch.no_grad():
            faces = self.model.detect(np.array(img))

        W,H = img.size
        objs = []
        for i,box in enumerate(faces):
            x1,y1,x2,y2,c = [int(v) for v in box.tolist()]
            x1,y1,x2,y2 = self.enlarge_face([x1,y1,x2,y2],W,H)
            mask = np.zeros([H,W]).astype(float)
            mask[y1:y2,x1:x2] = 1.0
            objs.append(dict(
                box=[x1,y1,x2,y2],
                category='face',
                inst_id=i,
                mask = mask
            ))
        return objs

    def html(self,img,output_var,objs):
        step_name = html_step_name(self.step_name)
        box_img = self.box_image(img, [obj['box'] for obj in objs])
        img = html_embed_image(img)
        box_img = html_embed_image(box_img,300)
        output_var = html_var_name(output_var)
        img_arg = html_arg_name('image')
        return f"""<div>{output_var}={step_name}({img_arg}={img})={box_img}</div>"""


    def execute(self,prog_step,inspect=False):
        img_var,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = self.det_face(img)
        prog_step.state[output_var] = objs
        if inspect:
            html_str = self.html(img, output_var, objs)
            return objs, html_str

        return objs


class EmojiInterpreter():
    step_name = 'EMOJI'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        emoji_name = eval(parse_result['args']['emoji'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,emoji_name,output_var

    def add_emoji(self,objs,emoji_name,img):
        W,H = img.size
        emojipth = os.path.join(EMOJI_DIR,f'smileys/{emoji_name}.png')
        for obj in objs:
            x1,y1,x2,y2 = obj['box']
            cx = (x1+x2)/2
            cy = (y1+y2)/2
            s = (y2-y1)/1.5
            x_pos = (cx-0.5*s)/W
            y_pos = (cy-0.5*s)/H
            emoji_size = s/H
            emoji_aug = imaugs.OverlayEmoji(
                emoji_path=emojipth,
                emoji_size=emoji_size,
                x_pos=x_pos,
                y_pos=y_pos)
            img = emoji_aug(img)

        return img

    def html(self,img_var,obj_var,emoji_name,output_var,img):
        step_name = html_step_name(self.step_name)
        image_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        emoji_arg = html_arg_name('emoji')
        image_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        img = html_embed_image(img,300)
        return f"""<div>{output_var}={step_name}({image_arg}={image_var},{obj_arg}={obj_var},{emoji_arg}='{emoji_name}')={img}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,emoji_name,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        img = self.add_emoji(objs, emoji_name, img)
        prog_step.state[output_var] = img
        if inspect:
            html_str = self.html(img_var, obj_var, emoji_name, output_var, img)
            return img, html_str

        return img

class ListInterpreter():
    step_name = 'LIST'

    prompt_template = """
Create comma separated lists based on the query.

Query: List at most 3 primary colors separated by commas
List:
red, blue, green

Query: List at most 2 north american states separated by commas
List:
California, Washington

Query: List at most {list_max} {text} separated by commas
List:"""

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        text = eval(parse_result['args']['query'])
        list_max = eval(parse_result['args']['max'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return text,list_max,output_var

    def get_list(self,text,list_max):
        response = client.completions.create(model="text-davinci-002",
        prompt=self.prompt_template.format(list_max=list_max,text=text),
        temperature=0.7,
        max_tokens=256,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=0,
        n=1)

        item_list = response.choices[0]['text'].lstrip('\n').rstrip('\n').split(', ')
        return item_list

    def html(self,text,list_max,item_list,output_var):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        query_arg = html_arg_name('query')
        max_arg = html_arg_name('max')
        output = html_output(item_list)
        return f"""<div>{output_var}={step_name}({query_arg}='{text}', {max_arg}={list_max})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        text,list_max,output_var = self.parse(prog_step)
        item_list = self.get_list(text,list_max)
        prog_step.state[output_var] = item_list
        if inspect:
            html_str = self.html(text, list_max, item_list, output_var)
            return item_list, html_str

        return item_list


class ClassifyInterpreter():
    step_name = 'CLASSIFY'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14").to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        image_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        category_var = parse_result['args']['categories']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return image_var,obj_var,category_var,output_var

    def calculate_sim(self,inputs):
        img_feats = self.model.get_image_features(inputs['pixel_values'])
        text_feats = self.model.get_text_features(inputs['input_ids'])
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        return torch.matmul(img_feats,text_feats.t())

    def query_obj(self,query,objs,img):
        if len(objs)==0:
            images = [img]
            return []
        else:
            images = [img.crop(obj['box']) for obj in objs]

        if len(query)==1:
            query = query + ['other']

        text = [f'a photo of {q}' for q in query]
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            sim = self.calculate_sim(inputs)


        # if only one query then select the object with the highest score
        if len(query)==1:
            scores = sim.cpu().numpy()
            obj_ids = scores.argmax(0)
            obj = objs[obj_ids[0]]
            obj['class']=query[0]
            obj['class_score'] = 100.0*scores[obj_ids[0],0]
            return [obj]

        # assign the highest scoring class to each object but this may assign same class to multiple objects
        scores = sim.cpu().numpy()
        cat_ids = scores.argmax(1)
        for i,(obj,cat_id) in enumerate(zip(objs,cat_ids)):
            class_name = query[cat_id]
            class_score = scores[i,cat_id]
            obj['class'] = class_name #+ f'({score_str})'
            obj['class_score'] = round(class_score*100,1)

        # sort by class scores and then for each class take the highest scoring object
        objs = sorted(objs,key=lambda x: x['class_score'],reverse=True)
        objs = [obj for obj in objs if 'class' in obj]
        classes = set([obj['class'] for obj in objs])
        new_objs = []
        for class_name in classes:
            cls_objs = [obj for obj in objs if obj['class']==class_name]

            max_score = 0
            max_obj = None
            for obj in cls_objs:
                if obj['class_score'] > max_score:
                    max_obj = obj
                    max_score = obj['class_score']

            new_objs.append(max_obj)

        return new_objs

    def html(self,img_var,obj_var,objs,cat_var,output_var):
        step_name = html_step_name(self.step_name)
        output = []
        for obj in objs:
            output.append(dict(
                box=obj['box'],
                tag=obj['class'],
                score=obj['class_score']
            ))
        output = html_output(output)
        output_var = html_var_name(output_var)
        img_var = html_var_name(img_var)
        cat_var = html_var_name(cat_var)
        obj_var = html_var_name(obj_var)
        img_arg = html_arg_name('image')
        cat_arg = html_arg_name('categories')
        return f"""<div>{output_var}={step_name}({img_arg}={img_var},{cat_arg}={cat_var})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        image_var,obj_var,category_var,output_var = self.parse(prog_step)
        img = prog_step.state[image_var]
        objs = prog_step.state[obj_var]
        cats = prog_step.state[category_var]
        objs = self.query_obj(cats, objs, img)
        prog_step.state[output_var] = objs
        if inspect:
            html_str = self.html(image_var,obj_var,objs,category_var,output_var)
            return objs, html_str

        return objs


class TagInterpreter():
    step_name = 'TAG'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,output_var

    def tag_image(self,img,objs):
        W,H = img.size
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 16)
        for i,obj in enumerate(objs):
            box = obj['box']
            draw.rectangle(box,outline='green',width=4)
            x1,y1,x2,y2 = box
            label = obj['class'] + '({})'.format(obj['class_score'])
            if 'class' in obj:
                w,h = font.getsize(label)
                if x1+w > W or y2+h > H:
                    draw.rectangle((x1, y2-h, x1 + w, y2), fill='green')
                    draw.text((x1,y2-h),label,fill='white',font=font)
                else:
                    draw.rectangle((x1, y2, x1 + w, y2 + h), fill='green')
                    draw.text((x1,y2),label,fill='white',font=font)
        return img1

    def html(self,img_var,tagged_img,obj_var,output_var):
        step_name = html_step_name(self.step_name)
        img_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        tagged_img = html_embed_image(tagged_img,300)
        img_arg = html_arg_name('image')
        obj_arg = html_arg_name('objects')
        output_var = html_var_name(output_var)
        return f"""<div>{output_var}={step_name}({img_arg}={img_var}, {obj_arg}={obj_var})={tagged_img}</div>"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,output_var = self.parse(prog_step)
        original_img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        img = self.tag_image(original_img, objs)
        prog_step.state[output_var] = img
        if inspect:
            html_str = self.html(img_var, img, obj_var, output_var)
            return img, html_str

        return img


def dummy(images, **kwargs):
    return images, False

class ReplaceInterpreter():
    step_name = 'REPLACE'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        device = "cuda"
        model_name = "runwayml/stable-diffusion-inpainting"
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_name,
            revision="fp16",
            torch_dtype=torch.float16)
        self.pipe = self.pipe.to(device)
        self.pipe.safety_checker = dummy

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_var = parse_result['args']['object']
        prompt = eval(parse_result['args']['prompt'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_var,prompt,output_var

    def create_mask_img(self,objs):
        mask = objs[0]['mask']
        mask[mask>0.5] = 255
        mask[mask<=0.5] = 0
        mask = mask.astype(np.uint8)
        return Image.fromarray(mask)

    def merge_images(self,old_img,new_img,mask):
        print(mask.size,old_img.size,new_img.size)

        mask = np.array(mask).astype(np.float)/255
        mask = np.tile(mask[:,:,np.newaxis],(1,1,3))
        img = mask*np.array(new_img) + (1-mask)*np.array(old_img)
        return Image.fromarray(img.astype(np.uint8))

    def resize_and_pad(self,img,size=(512,512)):
        new_img = Image.new(img.mode,size)
        thumbnail = img.copy()
        thumbnail.thumbnail(size)
        new_img.paste(thumbnail,(0,0))
        W,H = thumbnail.size
        return new_img, W, H

    def predict(self,img,mask,prompt):
        mask,_,_ = self.resize_and_pad(mask)
        init_img,W,H = self.resize_and_pad(img)
        new_img = self.pipe(
            prompt=prompt,
            image=init_img,
            mask_image=mask,
            # strength=0.98,
            guidance_scale=7.5,
            num_inference_steps=50 #200
        ).images[0]
        return new_img.crop((0,0,W-1,H-1)).resize(img.size)

    def html(self,img_var,obj_var,prompt,output_var,output):
        step_name = html_step_name(img_var)
        img_var = html_var_name(img_var)
        obj_var = html_var_name(obj_var)
        output_var = html_var_name(output_var)
        img_arg = html_arg_name('image')
        obj_arg = html_arg_name('object')
        prompt_arg = html_arg_name('prompt')
        output = html_embed_image(output,300)
        return f"""{output_var}={step_name}({img_arg}={img_var},{obj_arg}={obj_var},{prompt_arg}='{prompt}')={output}"""

    def execute(self,prog_step,inspect=False):
        img_var,obj_var,prompt,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        objs = prog_step.state[obj_var]
        mask = self.create_mask_img(objs)
        new_img = self.predict(img, mask, prompt)
        prog_step.state[output_var] = new_img
        if inspect:
            html_str = self.html(img_var, obj_var, prompt, output_var, new_img)
            return new_img, html_str
        return new_img

#### Clevr modules ####
class LocClevrInterpreter(LocInterpreter):
    def __init__(self, use_precomputed, thresh=0.1, nms_thresh=0.5):
        super().__init__(thresh, nms_thresh)
        with open(os.path.join(config['data_dir'], 'clevr', 'vocab_clevrer.json'), 'r') as f:
            vocab = json.load(f)
            obj2idx = vocab['object_name_to_idx']
        self.CLASS_NAMES = list(obj2idx.keys())
        self.use_precomputed = use_precomputed
        if self.use_precomputed:
            self.conn = duckdb.connect(database=os.path.join(config['db_dir'], 'annotations.duckdb'), read_only=True)
            # duckdb.execute("CREATE TABLE Obj_clevr (fid INT, oid INT, shape varchar, color varchar, material varchar, x1 float, y1 float, x2 float, y2 float)")
            # duckdb.execute("COPY Obj_clevr FROM '{}' (FORMAT 'csv', delimiter ',', header 0)".format(os.path.join(config["db_dir"], "obj_clevr.csv")))
        else:
            self.clevrer_model = torch.load(os.path.join(config['data_dir'], 'models', 'mask-rcnn-clevrer_epoch-44.pt'), map_location=torch.device('cpu'))
            self.clevrer_model.eval()
            self.clevrer_model.to(self.device)

    def predict(self,img,obj_name):
        encoding = self.processor(
            text=[[f'a photo of {obj_name}']],
            images=img,
            return_tensors='pt')
        encoding = {k:v.to(self.device) for k,v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            for k,v in outputs.items():
                if v is not None:
                    outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v

        target_sizes = torch.Tensor([img.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs,threshold=self.thresh,target_sizes=target_sizes)
        boxes, scores = results[0]["boxes"], results[0]["scores"]
        boxes = boxes.cpu().detach().numpy().tolist()
        scores = scores.cpu().detach().numpy().tolist()
        if len(boxes)==0:
            return []

        boxes, scores = zip(*sorted(zip(boxes,scores),key=lambda x: x[1],reverse=True))
        selected_boxes = []
        selected_scores = []
        for i in range(len(scores)):
            if scores[i] > self.thresh:
                coord = self.normalize_coord(boxes[i],img.size)
                selected_boxes.append(coord)
                selected_scores.append(scores[i])

        selected_boxes, selected_scores = nms(
            selected_boxes,selected_scores,self.nms_thresh)
        objs = []
        for i, (box, score) in enumerate(zip(selected_boxes, selected_scores)):
            objs.append(dict(
                box=box,
                score=score,
                category=obj_name,
                inst_id=i,
            ))
        return objs

    def predict_clevrer(self,img,obj_classes=None):
        with torch.no_grad():
            cv2_image = np.array(img)
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            x = transform(cv2_image).to(self.device)
            pred = self.clevrer_model([x, ])[0]
            indices = torchvision.ops.nms(pred["boxes"], pred["scores"], 0.3)
        boxes = pred["boxes"][indices]
        scores = pred["scores"][indices]
        labels = pred["labels"][indices]
        if obj_classes:
            # detect specific objects
            indices = []
            for i, label in enumerate(labels):
                # print(self.CLASS_NAMES[label])
                if set(obj_classes).issubset(self.CLASS_NAMES[label].split(' ')):
                    indices.append(i)
            boxes = boxes[indices].cpu().detach().numpy().tolist()
            scores = scores[indices].cpu().detach().numpy().tolist()
            labels = labels[indices].cpu().detach().numpy().tolist()
            if len(boxes)==0:
                return []
        else:
            boxes = boxes.cpu().detach().numpy().tolist()
            scores = scores.cpu().detach().numpy().tolist()
            labels = labels.cpu().detach().numpy().tolist()
        boxes = [self.normalize_coord(box,img.size) for box in boxes]
        objs = []
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            objs.append(dict(
                box=box,
                score=score,
                category=self.CLASS_NAMES[label],
                inst_id=i,
            ))
        return objs

    def predict_clevrer_precomputed(self,fid,obj_classes=None):
        # Obj_clevr (fid INT, oid INT, shape varchar, color varchar, material varchar, x1 float, y1 float, x2 float, y2 float)
        duckdb.execute("SELECT * FROM Obj_clevr WHERE fid = ?", (fid,))
        results = self.conn.fetchall()
        objs = []
        inst_id = 0
        for result in results:
            if obj_classes and set(obj_classes).issubset([result[3], result[4], result[2]]):
                objs.append(dict(
                    box=[result[5], result[6], result[7], result[8]],
                    score=1,
                    category=' '.join([result[3], result[4], result[2]]),
                    inst_id=inst_id,
                ))
                inst_id += 1
            elif not obj_classes:
                objs.append(dict(
                    box=[result[5], result[6], result[7], result[8]],
                    score=1,
                    category=' '.join([result[3], result[4], result[2]]),
                    inst_id=result[1],
                ))
        return objs

    def execute(self,prog_step,inspect=False):
        """
        Return bounding box
        """
        img_var,obj_name,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        if self.use_precomputed:
            fid = prog_step.state['fid']
            if obj_name=='object':
                # Run clevrer object detector to detect all objects
                objs = self.predict_clevrer_precomputed(fid)
            else:
                obj_classes = self.extract_words(obj_name)
                clevrer_attributes = ['cube', 'sphere', 'cylinder', 'metal', 'rubber', 'red', 'blue', 'green', 'yellow', 'purple', 'gray', 'brown', 'cyan']
                if set(obj_classes).issubset(clevrer_attributes):
                    # Run clevrer object detector for specific object classes
                    objs = self.predict_clevrer_precomputed(fid, obj_classes)
                else:
                    raise NotImplementedError("Precomputed bounding box only supports clevrer attributes")
        else:
            if obj_name=='object':
                # Run clevrer object detector to detect all objects
                objs = self.predict_clevrer(img)
            else:
                obj_classes = self.extract_words(obj_name)
                clevrer_attributes = ['cube', 'sphere', 'cylinder', 'metal', 'rubber', 'red', 'blue', 'green', 'yellow', 'purple', 'gray', 'brown', 'cyan']
                if set(obj_classes).issubset(clevrer_attributes):
                    # Run clevrer object detector for specific object classes
                    objs = self.predict_clevrer(img, obj_classes)
                else:
                    # Run owlvit model
                    objs = self.predict(img, obj_name) # can be empty list
        bboxes = [obj['box'] for obj in objs]
        box_img = self.box_image(img, bboxes)
        prog_step.state[output_var] = objs
        prog_step.state[output_var+'_IMAGE'] = box_img
        if inspect:
            html_str = self.html(img, box_img, output_var, obj_name)
            return objs, html_str

        return objs

class AttrClevrInterpreter():
    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        obj_var = parse_result['args']['object']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return obj_var,output_var

    def html(self, input_img, output_img, output_var):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        obj_arg = html_arg_name('object')
        obj_img = html_embed_image(input_img)
        output_img = html_embed_image(output_img)
        return f"""<div>{output_var}={step_name}({obj_arg}={obj_img})={output_img}</div>"""

    def box_image(self,img,boxes):
        img1 = img.copy()
        draw = ImageDraw.Draw(img1)
        for i,box in enumerate(boxes):
            draw.rectangle(box,outline='blue',width=5)

        return img1

    def execute(self,prog_step,inspect=False):
        obj_var,output_var = self.parse(prog_step)
        objs = prog_step.state[obj_var]
        output_objs = []
        for obj in objs:
            if self.step_name.lower() in obj['category'].split(' '):
                output_objs.append(obj)
        img = prog_step.state['IMAGE']
        output_img = self.box_image(img, [obj['box'] for obj in output_objs])
        prog_step.state[output_var] = output_objs
        prog_step.state[output_var+'_IMAGE'] = output_img
        if inspect:
            input_img = prog_step.state[obj_var+'_IMAGE']
            html_str = self.html(input_img, output_img, output_var)
            return output_objs, html_str

        return output_objs

class BigClevrInterpreter(AttrClevrInterpreter):
    step_name = 'BIG'

    def execute(self,prog_step,inspect=False):
        obj_var,output_var = self.parse(prog_step)
        objs = prog_step.state[obj_var]
        output_objs = []
        for obj in objs:
            area = (obj['box'][2]-obj['box'][0])*(obj['box'][3]-obj['box'][1])
            if area > 2400:
                output_objs.append(obj)
        img = prog_step.state['IMAGE']
        output_img = self.box_image(img, [obj['box'] for obj in output_objs])
        prog_step.state[output_var] = output_objs
        prog_step.state[output_var+'_IMAGE'] = output_img
        if inspect:
            input_img = prog_step.state[obj_var+'_IMAGE']
            html_str = self.html(input_img, output_img, output_var)
            return output_objs, html_str

        return output_objs

class SmallClevrInterpreter(AttrClevrInterpreter):
    step_name = 'SMALL'

    def execute(self,prog_step,inspect=False):
        obj_var,output_var = self.parse(prog_step)
        objs = prog_step.state[obj_var]
        output_objs = []
        for obj in objs:
            area = (obj['box'][2]-obj['box'][0])*(obj['box'][3]-obj['box'][1])
            if area <= 2400:
                output_objs.append(obj)
        img = prog_step.state['IMAGE']
        output_img = self.box_image(img, [obj['box'] for obj in output_objs])
        prog_step.state[output_var] = output_objs
        prog_step.state[output_var+'_IMAGE'] = output_img
        if inspect:
            input_img = prog_step.state[obj_var+'_IMAGE']
            html_str = self.html(input_img, output_img, output_var)
            return output_objs, html_str

        return output_objs

class GrayClevrInterpreter(AttrClevrInterpreter):
    step_name = 'GRAY'

class RedClevrInterpreter(AttrClevrInterpreter):
    step_name = 'RED'

class BlueClevrInterpreter(AttrClevrInterpreter):
    step_name = 'BLUE'

class GreenClevrInterpreter(AttrClevrInterpreter):
    step_name = 'GREEN'

class BrownClevrInterpreter(AttrClevrInterpreter):
    step_name = 'BROWN'

class PurpleClevrInterpreter(AttrClevrInterpreter):
    step_name = 'PURPLE'

class CyanClevrInterpreter(AttrClevrInterpreter):
    step_name = 'CYAN'

class YellowClevrInterpreter(AttrClevrInterpreter):
    step_name = 'YELLOW'

class CubeClevrInterpreter(AttrClevrInterpreter):
    step_name = 'CUBE'

class SphereClevrInterpreter(AttrClevrInterpreter):
    step_name = 'SPHERE'

class CylinderClevrInterpreter(AttrClevrInterpreter):
    step_name = 'CYLINDER'

class RubberClevrInterpreter(AttrClevrInterpreter):
    step_name = 'RUBBER'

class MetalClevrInterpreter(AttrClevrInterpreter):
    step_name = 'METAL'

class RelClevrInterpreter():
    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        object1_var = parse_result['args']['object1']
        object2_var = parse_result['args']['object2']
        assert(step_name==self.step_name)
        return object1_var, object2_var, output_var

    def html(self, object1_img, object2_img, output_var, rels):
        step_name = html_step_name(self.step_name)
        object1_arg=html_arg_name('object1')
        object2_arg=html_arg_name('object2')
        output_var = html_var_name(output_var)
        object1_img = html_embed_image(object1_img)
        object2_img = html_embed_image(object2_img)
        output = html_output_as_table(rels)
        return f"""<div>{output_var}={step_name}({object1_arg}={object1_img}, {object2_arg}={object2_img})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        object1_var, object2_var, output_var = self.parse(prog_step)
        object1 = prog_step.state[object1_var]
        object2 = prog_step.state[object2_var]
        rels = []
        for obj1 in object1:
            for obj2 in object2:
                if self.eval_rel(obj1, obj2):
                    rels.append(dict(
                        object1=obj1,
                        object2=obj2,
                        relation=self.step_name.lower()
                    ))
        prog_step.state[output_var] = rels
        if inspect:
            object1_img = prog_step.state[object1_var+'_IMAGE']
            object2_img = prog_step.state[object2_var+'_IMAGE']
            html_str = self.html(object1_img, object2_img, output_var, rels)
            return rels, html_str
        return rels

class LeftOfClevrInterpreter(RelClevrInterpreter):
    step_name = 'LEFTOF'
    def eval_rel(self, obj1, obj2):
        return obj1['inst_id'] != obj2['inst_id'] and obj1['box'][0] + obj1['box'][2] < obj2['box'][0] + obj2['box'][2]

class RightOfClevrInterpreter(RelClevrInterpreter):
    step_name = 'RIGHTOF'
    def eval_rel(self, obj1, obj2):
        return obj1['inst_id'] != obj2['inst_id'] and obj1['box'][0] + obj1['box'][2] > obj2['box'][0] + obj2['box'][2]

class FrontOfClevrInterpreter(RelClevrInterpreter):
    step_name = 'FRONTOF'
    def eval_rel(self, obj1, obj2):
        return obj1['inst_id'] != obj2['inst_id'] and obj1['box'][1] + obj1['box'][3] > obj2['box'][1] + obj2['box'][3]

class BehindClevrInterpreter(RelClevrInterpreter):
    step_name = 'BEHIND'
    def eval_rel(self, obj1, obj2):
        return obj1['inst_id'] != obj2['inst_id'] and obj1['box'][1] + obj1['box'][3] < obj2['box'][1] + obj2['box'][3]

class EqualSizeClevrInterpreter(RelClevrInterpreter):
    step_name = 'EQUALSIZE'
    def eval_rel(self, obj1, obj2):
        if obj1['inst_id'] == obj2['inst_id']:
            return False
        area1 = (obj1['box'][2]-obj1['box'][0])*(obj1['box'][3]-obj1['box'][1])
        area2 = (obj2['box'][2]-obj2['box'][0])*(obj2['box'][3]-obj2['box'][1])
        return (area1 > 2400 and area2 > 2400) or (area1 <= 2400 and area2 <= 2400)

class EqualMaterialClevrInterpreter(RelClevrInterpreter):
    step_name = 'EQUALMATERIAL'
    def eval_rel(self, obj1, obj2):
        if obj1['inst_id'] == obj2['inst_id']:
            return False
        return obj1['category'].split(' ')[1] == obj2['category'].split(' ')[1]

class EqualShapeClevrInterpreter(RelClevrInterpreter):
    step_name = 'EQUALSHAPE'
    def eval_rel(self, obj1, obj2):
        if obj1['inst_id'] == obj2['inst_id']:
            return False
        return obj1['category'].split(' ')[2] == obj2['category'].split(' ')[2]

class EqualColorClevrInterpreter(RelClevrInterpreter):
    step_name = 'EQUALCOLOR'
    def eval_rel(self, obj1, obj2):
        if obj1['inst_id'] == obj2['inst_id']:
            return False
        return obj1['category'].split(' ')[0] == obj2['category'].split(' ')[0]

#### Clevrer modules ####
class LocClevrerInterpreter(LocInterpreter):
    def __init__(self, use_precomputed):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        with open(os.path.join(config['data_dir'], 'clevr', 'vocab_clevrer.json'), 'r') as f:
            vocab = json.load(f)
            obj2idx = vocab['object_name_to_idx']
        self.CLASS_NAMES = list(obj2idx.keys())
        self.use_precomputed = use_precomputed
        if self.use_precomputed:
            self.conn = duckdb.connect(database=os.path.join(config['db_dir'], 'annotations.duckdb'), read_only=True)
            # duckdb.execute("CREATE TABLE Obj_clevrer (oid INT, vid INT, fid INT, shape varchar, color varchar, material varchar, x1 float, y1 float, x2 float, y2 float)")
            # duckdb.execute("COPY Obj_clevrer FROM '{}' (FORMAT 'csv', delimiter ',', header 0)".format(os.path.join(config["db_dir"], "obj_clevrer.csv")))
            # Creating index seems to produce incorrect results
            # duckdb.execute("CREATE INDEX IF NOT EXISTS idx_obj_clevrer ON Obj_clevrer (vid)")
        else:
            self.clevrer_model = torch.load(os.path.join(config['data_dir'], 'models', 'mask-rcnn-clevrer_epoch-44.pt'), map_location=torch.device('cpu'))
            self.clevrer_model.eval()
            self.clevrer_model.to(self.device)

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        video_var = parse_result['args']['video']
        obj_name = eval(parse_result['args']['object'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return video_var,obj_name,output_var

    def predict_clevrer(self,video,obj_classes=None):
        objs = []
        for fid, img in enumerate(video):
            with torch.no_grad():
                cv2_image = np.array(img)
                cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
                x = transform(cv2_image).to(self.device)
                pred = self.clevrer_model([x, ])[0]
                indices = torchvision.ops.nms(pred["boxes"], pred["scores"], 0.3)
            boxes = pred["boxes"][indices]
            labels = pred["labels"][indices]
            if obj_classes:
                # detect specific objects
                indices = []
                for i, label in enumerate(labels):
                    # print(self.CLASS_NAMES[label])
                    if set(obj_classes).issubset(self.CLASS_NAMES[label].split(' ')):
                        indices.append(i)
                boxes = boxes[indices].cpu().detach().numpy().tolist()
                labels = labels[indices].cpu().detach().numpy().tolist()
            else:
                boxes = boxes.cpu().detach().numpy().tolist()
                labels = labels.cpu().detach().numpy().tolist()
            boxes = [self.normalize_coord(box,img.size) for box in boxes]
            for i, (box, label) in enumerate(zip(boxes, labels)):
                color, material, shape = self.CLASS_NAMES[label].split(' ')
                objs.append(dict(
                    oid=i,
                    fid=fid,
                    x1=box[0],
                    y1=box[1],
                    x2=box[2],
                    y2=box[3],
                    shape=shape,
                    color=color,
                    material=material,
                ))
        return pd.DataFrame(objs)

    def predict_clevrer_precomputed(self,vid,obj_classes=None):
        # Obj_clevr (oid INT, vid INT, fid INT, shape varchar, color varchar, material varchar, x1 float, y1 float, x2 float, y2 float)
        # obj_clevrer_df = self.conn.execute("SELECT * FROM Obj_clevrer WHERE vid = ?", (vid,)).df()
        # print("obj_clevrer_df", obj_clevrer_df.loc[obj_clevrer_df['fid'] == 0])
        df = self.conn.execute("SELECT * EXCLUDE (vid) FROM Obj_clevrer WHERE vid = ?", (vid,)).df()
        if obj_classes:
            raise NotImplementedError("only object='object' is supported for precomputed bounding box")
        return df

    def execute(self,prog_step,inspect=False):
        video_var,obj_name,output_var = self.parse(prog_step)
        video = prog_step.state[video_var]
        if self.use_precomputed:
            vid = prog_step.state['vid']
            if obj_name=='object':
                # Run clevrer object detector to detect all objects
                df = self.predict_clevrer_precomputed(vid)
            else:
                obj_classes = self.extract_words(obj_name)
                clevrer_attributes = ['cube', 'sphere', 'cylinder', 'metal', 'rubber', 'red', 'blue', 'green', 'yellow', 'purple', 'gray', 'brown', 'cyan']
                if set(obj_classes).issubset(clevrer_attributes):
                    # Run clevrer object detector for specific object classes
                    df = self.predict_clevrer_precomputed(vid, obj_classes)
                else:
                    raise NotImplementedError("Precomputed bounding box only supports clevrer attributes")
        else:
            if obj_name=='object':
                # Run clevrer object detector to detect all objects
                df = self.predict_clevrer(video)
            else:
                obj_classes = self.extract_words(obj_name)
                clevrer_attributes = ['cube', 'sphere', 'cylinder', 'metal', 'rubber', 'red', 'blue', 'green', 'yellow', 'purple', 'gray', 'brown', 'cyan']
                if set(obj_classes).issubset(clevrer_attributes):
                    # Run clevrer object detector for specific object classes
                    df = self.predict_clevrer(video, obj_classes)
                else:
                    raise NotImplementedError("only supports clevrer attributes")

        prog_step.state[output_var] = df
        if inspect:
            raise NotImplementedError("inspect not supported for clevrer")
        # print(f"{self.step_name} dataframe ", df.loc[df['fid']==0])
        return df

class TrackClevrerInterpreter():
    step_name = 'TRACK'

    def __init__(self, use_precomputed):
        print(f'Registering {self.step_name} step')
        self.use_precomputed = use_precomputed
        if not self.use_precomputed:
            self.args = Namespace(
                img_info=[320, 480],
                track_thresh=0.5,
                track_buffer=30,
                match_thresh=0.8,
                min_box_area=0,
                aspect_ratio_thresh=1000,
                mot20=False
            )
            self.tracker = BYTETracker(self.args)

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        obj_var = parse_result['args']['object']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return obj_var,output_var

    def execute(self,prog_step,inspect=False):
        obj_var,output_var = self.parse(prog_step)
        obj_df = prog_step.state[obj_var]
        if self.use_precomputed:
            output_df = obj_df
        else:
            # object tracking with ByteTrack
            output = []
            obj_df['score'] = 1
            for fid, df_fid in obj_df.groupby('fid'):
                dets = df_fid[['x1', 'y1', 'x2', 'y2', 'score', 'color', 'material', 'shape']].values
                if dets is not None:
                    online_targets = self.tracker.update(
                        dets,
                        self.args.img_info,
                        self.args.img_info
                    )
                    for t in online_targets:
                        tlwh = t._original_tlwh
                        tid = t.track_id
                        attrs = t.attrs
                        vertical = tlwh[2] / tlwh[3] > self.args.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                            output.append([fid, tid, tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3], *attrs])
            output_df = pd.DataFrame(output, columns=['fid', 'oid', 'x1', 'y1', 'x2', 'y2', 'color', 'material', 'shape'])
            # TODO: Store output to database
        prog_step.state[output_var] = output_df
        if inspect:
            raise NotImplementedError("inspect not supported for clevrer")
        # print(f"{self.step_name} dataframe ", df.loc[df['fid']==0])
        return output_df

class AttrClevrerInterpreter():
    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        obj_var = parse_result['args']['object']
        bind_variable = eval(parse_result['args']['var'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return obj_var, bind_variable, output_var

    def execute(self,prog_step,inspect=False):
        obj_var, bind_variable, output_var = self.parse(prog_step)
        # obj_df: (oid, fid, shape, color, material, x1, y1, x2, y2)
        obj_df = prog_step.state[obj_var]
        output_df = duckdb.execute("""
            SELECT oid as {}_oid, fid
            FROM obj_df
            WHERE {} = '{}'
        """.format(bind_variable, self.col_name, self.step_name.lower())).df()
        prog_step.state[output_var] = output_df
        if inspect:
            raise NotImplementedError("inspect not supported for clevrer")
        # print(f"{self.step_name} dataframe ", output_df.loc[output_df['fid'] == 0])
        return output_df

class LeftClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'LEFT'

    def execute(self,prog_step,inspect=False):
        obj_var, bind_variable, output_var = self.parse(prog_step)
        # obj_df: (oid, fid, shape, color, material, x1, y1, x2, y2)
        obj_df = prog_step.state[obj_var]
        # print("obj_df", obj_df.loc[obj_df['oid'] == 6])
        output_df = duckdb.execute("SELECT oid as {}_oid, fid FROM obj_df WHERE Left(x1, x2) = true".format(bind_variable)).df()
        prog_step.state[output_var] = output_df
        if inspect:
            raise NotImplementedError("inspect not supported for clevrer")
        # if bind_variable == 'o2':
        #     print(f"{self.step_name} dataframe ", output_df.loc[output_df['o2_oid'] == 6])
        return output_df

class RightClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'RIGHT'

    def execute(self,prog_step,inspect=False):
        obj_var, bind_variable, output_var = self.parse(prog_step)
        # obj_df: (oid, fid, shape, color, material, x1, y1, x2, y2)
        obj_df = prog_step.state[obj_var]
        output_df = duckdb.execute("SELECT oid as {}_oid, fid FROM obj_df WHERE Right(x1, x2) = true".format(bind_variable)).df()
        prog_step.state[output_var] = output_df
        if inspect:
            raise NotImplementedError("inspect not supported for clevrer")
        # print(f"{self.step_name} dataframe ", output_df.loc[output_df['o2_oid'] == 6])
        return output_df

class TopClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'TOP'

    def execute(self,prog_step,inspect=False):
        obj_var, bind_variable, output_var = self.parse(prog_step)
        # obj_df: (oid, fid, shape, color, material, x1, y1, x2, y2)
        obj_df = prog_step.state[obj_var]
        output_df = duckdb.execute("SELECT oid as {}_oid, fid FROM obj_df WHERE Top(y1, y2) = true".format(bind_variable)).df()
        prog_step.state[output_var] = output_df
        if inspect:
            raise NotImplementedError("inspect not supported for clevrer")
        # if bind_variable == 'o2':
        #     print(f"{self.step_name} dataframe ", output_df.loc[output_df['o2_oid'] == 6])
        return output_df

class BottomClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'BOTTOM'

    def execute(self,prog_step,inspect=False):
        obj_var, bind_variable, output_var = self.parse(prog_step)
        # obj_df: (oid, fid, shape, color, material, x1, y1, x2, y2)
        obj_df = prog_step.state[obj_var]
        output_df = duckdb.execute("SELECT oid as {}_oid, fid FROM obj_df WHERE Bottom(y1, y2) = true".format(bind_variable)).df()
        prog_step.state[output_var] = output_df
        if inspect:
            raise NotImplementedError("inspect not supported for clevrer")
        # print(f"{self.step_name} dataframe ", output_df)
        return output_df

class GrayClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'GRAY'
    col_name = 'color'

class RedClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'RED'
    col_name = 'color'

class BlueClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'BLUE'
    col_name = 'color'

class GreenClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'GREEN'
    col_name = 'color'

class BrownClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'BROWN'
    col_name = 'color'

class PurpleClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'PURPLE'
    col_name = 'color'

class CyanClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'CYAN'
    col_name = 'color'

class YellowClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'YELLOW'
    col_name = 'color'

class CubeClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'CUBE'
    col_name = 'shape'

class SphereClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'SPHERE'
    col_name = 'shape'

class CylinderClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'CYLINDER'
    col_name = 'shape'

class RubberClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'RUBBER'
    col_name = 'material'

class MetalClevrerInterpreter(AttrClevrerInterpreter):
    step_name = 'METAL'
    col_name = 'material'

class RelClevrerInterpreter():
    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        object1_var = parse_result['args']['object1']
        object2_var = parse_result['args']['object2']
        bind_variable1 = eval(parse_result['args']['var1'])
        bind_variable2 = eval(parse_result['args']['var2'])
        assert(step_name==self.step_name)
        return object1_var, object2_var, bind_variable1, bind_variable2, output_var

class LeftOfClevrerInterpreter(RelClevrerInterpreter):
    step_name = 'LEFTOF'
    def execute(self,prog_step,inspect=False):
        object1_var, object2_var, bind_variable1, bind_variable2, output_var = self.parse(prog_step)
        obj1_df = prog_step.state[object1_var]
        obj2_df = prog_step.state[object2_var]
        output_df = duckdb.execute("""
            SELECT obj1_df.oid AS {}_oid, obj2_df.oid AS {}_oid, obj1_df.fid AS fid
            FROM obj1_df, obj2_df
            WHERE obj1_df.fid = obj2_df.fid AND obj1_df.oid <> obj2_df.oid
                AND LeftOf(obj1_df.x1, obj1_df.x2, obj2_df.x1, obj2_df.x2) = true
        """.format(bind_variable1, bind_variable2)).df()
        prog_step.state[output_var] = output_df
        if inspect:
            raise NotImplementedError("inspect not supported for clevrer")
        # print(f"{self.step_name} dataframe ", output_df)
        return output_df

class RightOfClevrerInterpreter(RelClevrerInterpreter):
    step_name = 'RIGHTOF'
    def execute(self,prog_step,inspect=False):
        object1_var, object2_var, bind_variable1, bind_variable2, output_var = self.parse(prog_step)
        obj1_df = prog_step.state[object1_var]
        obj2_df = prog_step.state[object2_var]
        output_df = duckdb.execute("""
            SELECT obj1_df.oid AS {}_oid, obj2_df.oid AS {}_oid, obj1_df.fid AS fid
            FROM obj1_df, obj2_df
            WHERE obj1_df.fid = obj2_df.fid AND obj1_df.oid <> obj2_df.oid
                AND RightOf(obj1_df.x1, obj1_df.x2, obj2_df.x1, obj2_df.x2) = true
        """.format(bind_variable1, bind_variable2)).df()
        prog_step.state[output_var] = output_df
        if inspect:
            raise NotImplementedError("inspect not supported for clevrer")
        # print(f"{self.step_name} dataframe ", output_df)
        return output_df

class FrontOfClevrerInterpreter(RelClevrerInterpreter):
    step_name = 'FRONTOF'
    def execute(self,prog_step,inspect=False):
        object1_var, object2_var, bind_variable1, bind_variable2, output_var = self.parse(prog_step)
        obj1_df = prog_step.state[object1_var]
        obj2_df = prog_step.state[object2_var]
        output_df = duckdb.execute("""
            SELECT obj1_df.oid AS {}_oid, obj2_df.oid AS {}_oid, obj1_df.fid AS fid
            FROM obj1_df, obj2_df
            WHERE obj1_df.fid = obj2_df.fid AND obj1_df.oid <> obj2_df.oid
                AND FrontOf(obj1_df.y1, obj1_df.y2, obj2_df.y1, obj2_df.y2) = true
        """.format(bind_variable1, bind_variable2)).df()
        prog_step.state[output_var] = output_df
        if inspect:
            raise NotImplementedError("inspect not supported for clevrer")
        # print(f"{self.step_name} dataframe ", output_df)
        return output_df

class BehindClevrerInterpreter(RelClevrerInterpreter):
    step_name = 'BEHIND'
    def execute(self,prog_step,inspect=False):
        object1_var, object2_var, bind_variable1, bind_variable2, output_var = self.parse(prog_step)
        obj1_df = prog_step.state[object1_var]
        obj2_df = prog_step.state[object2_var]
        output_df = duckdb.execute("""
            SELECT obj1_df.oid AS {}_oid, obj2_df.oid AS {}_oid, obj1_df.fid AS fid
            FROM obj1_df, obj2_df
            WHERE obj1_df.fid = obj2_df.fid AND obj1_df.oid <> obj2_df.oid
                AND Behind(obj1_df.y1, obj1_df.y2, obj2_df.y1, obj2_df.y2) = true
        """.format(bind_variable1, bind_variable2)).df()
        prog_step.state[output_var] = output_df
        if inspect:
            raise NotImplementedError("inspect not supported for clevrer")
        # print(f"{self.step_name} dataframe ", output_df)
        return output_df

class EventClevrerInterpreter():
    step_name = 'EVENT'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        predicates_vars = parse_result['args']['predicates'][1:-1].split(',')
        min_duration = eval(parse_result['args']['min_duration'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return predicates_vars, min_duration, output_var

    def execute(self,prog_step,inspect=False):
        predicates_vars, min_duration, output_var = self.parse(prog_step)
        # List of dataframes for each predicate
        predicate_dfs = [prog_step.state[predicates_var] for predicates_var in predicates_vars]
        # Bind each dataframe to a variable name, so that we can refer to them in the SQL query
        predicate_df_names = [f"pred{i}_df" for i in range(len(predicate_dfs))]
        for predicate_df, predicate_df_name in zip(predicate_dfs, predicate_df_names):
            # print(f"{predicate_df_name} dataframe ", predicate_df.to_string())
            exec(f"{predicate_df_name} = predicate_df")
        # Compute natural join of dataframes. It's guaranteed that there are common columns for each pair of dataframes, as each dataframe has fid column.
        from_join_clause = " natural join ".join(predicate_df_names)
        graph_df = duckdb.execute(f"SELECT * FROM {from_join_clause}").df()
        # print("from_join_clause", from_join_clause)
        # if 'o2_oid' in graph_df.columns:
            # print("graph_df", graph_df.loc[graph_df['o2_oid'] == 6])

        # Filter events that last for at least min_duration frames. As an optimization, only events with exactly min_duration frames are retained.
        oid_cols = [col_name for col_name in graph_df.columns if col_name.endswith('_oid')]
        oid_cols = ", ".join(oid_cols)
        output_df = duckdb.execute(f"""
            SELECT {oid_cols}, fid AS start_fid, fid_offset AS end_fid
            FROM (
                SELECT {oid_cols},
                    fid,
                    lead(fid, {min_duration} - 1, 0) OVER (PARTITION BY {oid_cols} ORDER BY fid) as fid_offset
                FROM graph_df
            ) t
            WHERE fid_offset = fid + ({min_duration} - 1)
        """).df()
        prog_step.state[output_var] = output_df
        if inspect:
            raise NotImplementedError("inspect not supported for clevrer")
        # print(f"{self.step_name} dataframe ", output_df)
        return output_df

class BeforeClevrerInterpreter():
    step_name = 'BEFORE'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        event1_var = parse_result['args']['event1']
        event2_var = parse_result['args']['event2']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return event1_var, event2_var, output_var

    def execute(self,prog_step,inspect=False):
        event1_var, event2_var, output_var = self.parse(prog_step)
        event1_df = prog_step.state[event1_var]
        event2_df = prog_step.state[event2_var]
        # print("event1_df", event1_df.columns)
        # print("event2_df", event2_df.columns)
        event1_obj_cols = [col_name for col_name in event1_df.columns if col_name.endswith('_oid')]
        event2_obj_cols = [col_name for col_name in event2_df.columns if col_name.endswith('_oid')]
        obj_intersection_fields = []
        obj_union_fields = []
        for v in event1_obj_cols:
            obj_union_fields.append(f"t1.{v}")
        for v in event2_obj_cols:
            if v in event1_obj_cols:
                obj_intersection_fields.append(f"t1.{v} = t2.{v}")
            else:
                for u in event1_obj_cols:
                    obj_intersection_fields.append(f"t1.{u} <> t2.{v}")
                obj_union_fields.append(f"t2.{v}")
        obj_intersection_fields = " AND ".join(obj_intersection_fields)
        obj_union_fields = ", ".join(obj_union_fields)
        # As an optimization, only the earliest matching event is retained for each group of objects.
        output_df = duckdb.execute(f"""
            SELECT {obj_union_fields},
                min(t1.start_fid) AS start_fid,
                min(t2.end_fid) AS end_fid
            FROM event1_df t1, event2_df t2
            WHERE {obj_intersection_fields}
                AND t1.end_fid < t2.start_fid
            GROUP BY {obj_union_fields}
        """).df()
        prog_step.state[output_var] = output_df
        if inspect:
            raise NotImplementedError("inspect not supported for clevrer")
        # print(f"{self.step_name} dataframe ", output_df)
        return output_df

#### Video counterparts ####
class LocVideoInterpreter(LocInterpreter):
    def __init__(self,thresh=0.1,nms_thresh=0.5):
        super().__init__(thresh,nms_thresh)
        self.clevrer_model = torch.load(os.path.join(config['data_dir'], 'models', 'mask-rcnn-clevrer_epoch-44.pt'), map_location=torch.device('cpu'))
        self.clevrer_model.eval()
        self.clevrer_model.to(self.device)

        with open(os.path.join(config['data_dir'], "clevr", "vocab_clevrer.json"), 'r') as f:
            vocab = json.load(f)
            obj2idx = vocab['object_name_to_idx']
        self.CLASS_NAMES = list(obj2idx.keys())

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        video_var = parse_result['args']['video']
        obj_name = eval(parse_result['args']['object'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return video_var,obj_name,output_var

    def predict_clevrer(self,img,obj_classes):
        with torch.no_grad():
            cv2_image = np.array(img)
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            x = transform(cv2_image).to(self.device)
            pred = self.clevrer_model([x, ])[0]
            indices = torchvision.ops.nms(pred["boxes"], pred["scores"], 0.3)
        boxes = pred["boxes"][indices]
        scores = pred["scores"][indices]
        labels = pred["labels"][indices]
        indices = []
        for i, label in enumerate(labels):
            # print(self.CLASS_NAMES[label])
            if set(obj_classes).issubset(self.CLASS_NAMES[label].split(' ')):
                indices.append(i)
        boxes = boxes[indices].cpu().detach().numpy().tolist()
        scores = scores[indices].cpu().detach().numpy().tolist()
        if len(boxes)==0:
            return []
        boxes = [self.normalize_coord(box,img.size) for box in boxes]
        return boxes

    def create_box_video(self, video, fid_bbox_dict, highlight_best=True):
        box_video = {}
        for fid, img in video.items():
            box_img = self.box_image(img, fid_bbox_dict[fid], highlight_best)
            box_video[fid] = box_img
        return box_video

    def html(self,video,box_video,output_var,obj_name):
        step_name=html_step_name(self.step_name)
        obj_arg=html_arg_name('object')
        video_arg=html_arg_name('video')
        output_var=html_var_name(output_var)
        video=html_embed_video(video)
        box_video=html_embed_video(box_video,300)
        return f"<div>{output_var}={step_name}({video_arg}={video}, {obj_arg}='{obj_name}')={box_video}</div>"

    def execute(self,prog_step,inspect=False):
        """
        Return bounding box
        """
        video_var,obj_name,output_var = self.parse(prog_step)
        video = prog_step.state[video_var] # video: {fid: fid, image: PIL image}
        if obj_name=='TOP':
            fid_bbox_dict = {fid: self.top_box(img) for fid, img in video.items()}
        elif obj_name=='BOTTOM':
            fid_bbox_dict = {fid: self.bottom_box(img) for fid, img in video.items()}
        elif obj_name=='LEFT':
            fid_bbox_dict = {fid: self.left_box(img) for fid, img in video.items()}
        elif obj_name=='RIGHT':
            fid_bbox_dict = {fid: self.right_box(img) for fid, img in video.items()}
        else:
            obj_classes = self.extract_words(obj_name)
            clevrer_attributes = ['cube', 'sphere', 'cylinder', 'metal', 'rubber', 'red', 'blue', 'green', 'yellow', 'purple', 'gray', 'brown', 'cyan']
            if set(obj_classes).issubset(clevrer_attributes):
                # Run clevrer object detector
                fid_bbox_dict = {}
                for fid, img in tqdm(video.items()):
                    bboxes = self.predict_clevrer(img, obj_classes) # can be empty list
                    fid_bbox_dict[fid] = bboxes
            else:
                # Run owlvit model
                fid_bbox_dict = {}
                for fid, img in tqdm(video.items()):
                    bboxes = self.predict(img, obj_name) # can be empty list
                    fid_bbox_dict[fid] = bboxes

        box_video = self.create_box_video(video, fid_bbox_dict)
        prog_step.state[output_var] = fid_bbox_dict
        prog_step.state[output_var+'_VIDEO'] = box_video
        if inspect:
            html_str = self.html(video, box_video, output_var, obj_name)
            return fid_bbox_dict, html_str

        return fid_bbox_dict

class RelDetVideoInterpreter():
    # REL0=RELDET(box1=BOX1,box2=BOX2,relation='leftof')
    # REL0: {fid, List[Dict(box1, box2, relation)]}, where fid is the union of fid in BOX1 and BOX2
    step_name = 'RELDET'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        frames1_var = parse_result['args']['frames1']
        frames2_var = parse_result['args']['frames2']
        assert(step_name==self.step_name)
        return frames1_var, frames2_var, output_var

    def html(self, frames1, frames2, output_var, step_output):
        step_name = html_step_name(self.step_name)
        frames1_arg=html_arg_name('frames1')
        frames2_arg=html_arg_name('frames2')
        output_var = html_var_name(output_var)
        frames1 = html_output(frames1)
        frames2 = html_output(frames2)
        output = html_output(step_output)
        return f"""<div>{output_var}={step_name}({frames1_arg}={frames1}, {frames2_arg}={frames2})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        frames1_var, frames2_var, output_var = self.parse(prog_step)
        frames1 = prog_step.state[frames1_var]
        frames2 = prog_step.state[frames2_var]
        if frames1 is None or frames1 == [] or frames2 is None or frames2 == []:
            step_output = 'no'
        elif any(min(frames1) < v for v in frames2):
            step_output = 'yes'
        else:
            step_output = 'no'
        prog_step.state[output_var] = step_output
        if inspect:
            html_str = self.html(frames1, frames2, output_var, step_output)
            return step_output, html_str

        return step_output

class CropVideoInterpreter():
    step_name = 'CROP'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def expand_box(self,box,img_size,factor=1.5):
        W,H = img_size
        x1,y1,x2,y2 = box
        dw = int(factor*(x2-x1)/2)
        dh = int(factor*(y2-y1)/2)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        x1 = max(0,cx - dw)
        x2 = min(cx + dw,W)
        y1 = max(0,cy - dh)
        y2 = min(cy + dh,H)
        return [x1,y1,x2,y2]

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        video_var = parse_result['args']['video']
        box_var = parse_result['args']['box']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return video_var,box_var,output_var

    def html(self, out_video, output_var, box_video):
        # video = html_embed_video(video)
        out_video = html_embed_video(out_video,300)
        box_video = html_embed_video(box_video)
        output_var = html_var_name(output_var)
        step_name = html_step_name(self.step_name)
        box_arg = html_arg_name('bbox')
        return f"""<div>{output_var}={step_name}({box_arg}={box_video})={out_video}</div>"""

    def execute(self,prog_step,inspect=False):
        video_var,box_var,output_var = self.parse(prog_step)
        video = prog_step.state[video_var]
        boxes = prog_step.state[box_var]
        out_video = {}
        # For each video frame, crop by at most one bounding box
        for fid, bboxes_per_frame in boxes.items():
            if len(bboxes_per_frame) > 0:
                box = bboxes_per_frame[0]
                box = self.expand_box(box, video[fid].size)
                out_img = video[fid].crop(box)
            else:
                box = []
                out_img = video[fid]
            out_video[fid] = out_img

        prog_step.state[output_var] = out_video
        if inspect:
            box_video = prog_step.state[box_var+'_VIDEO']
            html_str = self.html(out_video, output_var, box_video)
            return out_video, html_str

        return out_video

class CropRightOfVideoInterpreter(CropVideoInterpreter, CropRightOfInterpreter):
    step_name = 'CROP_RIGHTOF'

    def execute(self,prog_step,inspect=False):
        video_var,box_var,output_var = self.parse(prog_step)
        video = prog_step.state[video_var]
        boxes = prog_step.state[box_var]
        out_video = {}
        # For each video frame, crop by at most one bounding box
        for fid, bboxes_per_frame in boxes.items():
            if len(bboxes_per_frame) > 0:
                box = bboxes_per_frame[0]
                right_box = self.right_of(box, video[fid].size)
            else:
                w,h = video[fid].size
                box = []
                right_box = [int(w/2),0,w-1,h-1]
            out_img = video[fid].crop(right_box)
            out_video[fid] = out_img

        prog_step.state[output_var] = out_video
        if inspect:
            box_video = prog_step.state[box_var+'_VIDEO']
            html_str = self.html(out_video, output_var, box_video)
            return out_video, html_str

        return out_video

class CropLeftOfVideoInterpreter(CropVideoInterpreter, CropLeftOfInterpreter):
    step_name = 'CROP_LEFTOF'

    def execute(self,prog_step,inspect=False):
        video_var,box_var,output_var = self.parse(prog_step)
        video = prog_step.state[video_var]
        boxes = prog_step.state[box_var]
        out_video = {}
        # For each video frame, crop by at most one bounding box
        for fid, bboxes_per_frame in boxes.items():
            if len(bboxes_per_frame) > 0:
                box = bboxes_per_frame[0]
                left_box = self.left_of(box, video[fid].size)
            else:
                w,h = video[fid].size
                box = []
                left_box = [0,0,int(w/2),h-1]
            out_img = video[fid].crop(left_box)
            out_video[fid] = out_img

        prog_step.state[output_var] = out_video
        if inspect:
            box_video = prog_step.state[box_var+'_VIDEO']
            html_str = self.html(out_video, output_var, box_video)
            return out_video, html_str

        return out_video

class CropAboveVideoInterpreter(CropVideoInterpreter, CropAboveInterpreter):
    step_name = 'CROP_ABOVE'

    def execute(self,prog_step,inspect=False):
        video_var,box_var,output_var = self.parse(prog_step)
        video = prog_step.state[video_var]
        boxes = prog_step.state[box_var]
        out_video = {}
        # For each video frame, crop by at most one bounding box
        for fid, bboxes_per_frame in boxes.items():
            if len(bboxes_per_frame) > 0:
                box = bboxes_per_frame[0]
                above_box = self.above(box, video[fid].size)
            else:
                w,h = video[fid].size
                box = []
                above_box = [0,0,int(w/2),h-1]
            out_img = video[fid].crop(above_box)
            out_video[fid] = out_img

        prog_step.state[output_var] = out_video
        if inspect:
            box_video = prog_step.state[box_var+'_VIDEO']
            html_str = self.html(out_video, output_var, box_video)
            return out_video, html_str

        return out_video

class CropBelowVideoInterpreter(CropVideoInterpreter, CropBelowInterpreter):
    step_name = 'CROP_BELOW'

    def execute(self,prog_step,inspect=False):
        video_var,box_var,output_var = self.parse(prog_step)
        video = prog_step.state[video_var]
        boxes = prog_step.state[box_var]
        out_video = {}
        # For each video frame, crop by at most one bounding box
        for fid, bboxes_per_frame in boxes.items():
            if len(bboxes_per_frame) > 0:
                box = bboxes_per_frame[0]
                below_box = self.below(box, video[fid].size)
            else:
                w,h = video[fid].size
                box = []
                below_box = [0,0,int(w/2),h-1]
            out_img = video[fid].crop(below_box)
            out_video[fid] = out_img

        prog_step.state[output_var] = out_video
        if inspect:
            box_video = prog_step.state[box_var+'_VIDEO']
            html_str = self.html(out_video, output_var, box_video)
            return out_video, html_str

        return out_video

class CropFrontOfVideoInterpreter(CropBelowVideoInterpreter):
    step_name = 'CROP_FRONTOF'

class CropInFrontVideoInterpreter(CropBelowVideoInterpreter):
    step_name = 'CROP_INFRONT'

class CropInFrontOfVideoInterpreter(CropBelowVideoInterpreter):
    step_name = 'CROP_INFRONTOF'

class CropBehindVideoInterpreter(CropAboveVideoInterpreter):
    step_name = 'CROP_BEHIND'

class CropAheadVideoInterpreter(CropBelowVideoInterpreter):
    step_name = 'CROP_AHEAD'

class CountVideoInterpreter(CountInterpreter):
    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        box_var = parse_result['args']['box']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return box_var,output_var

    def html(self,box_video, output_var, count_dict):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        box_arg = html_arg_name('bbox')
        box_video = html_embed_video(box_video)
        output = html_output_as_table(count_dict)
        return f"""<div>{output_var}={step_name}({box_arg}={box_video})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        box_var,output_var = self.parse(prog_step)
        boxes = prog_step.state[box_var]
        count_dict = {}
        for fid, bboxes_per_frame in boxes.items():
            count_dict[fid] = len(bboxes_per_frame)
        prog_step.state[output_var] = count_dict
        if inspect:
            box_video = prog_step.state[box_var+'_VIDEO']
            html_str = self.html(box_video, output_var, count_dict)
            return count_dict, html_str

        return count_dict

class EvalVideoInterpreter(EvalInterpreter):
    step_name = 'EVAL'

class ResultVideoInterpreter(ResultInterpreter):
    def html(self,output,output_var):
        step_name = html_step_name(self.step_name)
        output_var = html_var_name(output_var)
        if isinstance(output, Image.Image):
            output = html_embed_image(output,300)
        elif isinstance(output, dict):
            output = html_embed_video(output,300)
        else:
            output = html_output(output)

        return f"""<div>{step_name} -> {output_var} -> {output}</div>"""

class BeforeVideoInterpreter():
    step_name = 'BEFORE'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        frames1_var = parse_result['args']['frames1']
        frames2_var = parse_result['args']['frames2']
        assert(step_name==self.step_name)
        return frames1_var, frames2_var, output_var

    def html(self, frames1, frames2, output_var, step_output):
        step_name = html_step_name(self.step_name)
        frames1_arg=html_arg_name('frames1')
        frames2_arg=html_arg_name('frames2')
        output_var = html_var_name(output_var)
        frames1 = html_output(frames1)
        frames2 = html_output(frames2)
        output = html_output(step_output)
        return f"""<div>{output_var}={step_name}({frames1_arg}={frames1}, {frames2_arg}={frames2})={output}</div>"""

    def execute(self,prog_step,inspect=False):
        frames1_var, frames2_var, output_var = self.parse(prog_step)
        frames1 = prog_step.state[frames1_var]
        frames2 = prog_step.state[frames2_var]
        if frames1 is None or frames1 == [] or frames2 is None or frames2 == []:
            step_output = 'no'
        elif any(min(frames1) < v for v in frames2):
            step_output = 'yes'
        else:
            step_output = 'no'
        prog_step.state[output_var] = step_output
        if inspect:
            html_str = self.html(frames1, frames2, output_var, step_output)
            return step_output, html_str

        return step_output

class AfterVideoInterpreter():
    step_name = 'AFTER'

    def execute(self,prog_step,inspect=False):
        frames1_var, frames2_var, output_var = self.parse(prog_step)
        frames1 = prog_step.state[frames1_var]
        frames2 = prog_step.state[frames2_var]
        if frames1 is None or frames1 == [] or frames2 is None or frames2 == []:
            step_output = 'no'
        elif any(min(frames2) < v for v in frames1):
            step_output = 'yes'
        else:
            step_output = 'no'
        prog_step.state[output_var] = step_output
        if inspect:
            html_str = self.html(frames1, frames2, output_var, step_output)
            return step_output, html_str

        return step_output

#### User-defined functions ####
# TODO: take a look at PyArrow UDFs
def left(o1_x1: float, o1_x2: float) -> bool:
    cx1 = (o1_x1 + o1_x2) / 2
    return cx1 >= 0 and cx1 < 240

def right(o1_x1: float, o1_x2: float) -> bool:
    cx1 = (o1_x1 + o1_x2) / 2
    return cx1 >= 240 and cx1 <= 480

def top(o1_y1: float, o1_y2: float) -> bool:
    cy1 = (o1_y1 + o1_y2) / 2
    return cy1 >= 0 and cy1 < 160

def bottom(o1_y1: float, o1_y2: float) -> bool:
    cy1 = (o1_y1 + o1_y2) / 2
    return cy1 >= 160 and cy1 <= 320

def right_of(o1_x1: float, o1_x2: float, o2_x1: float, o2_x2: float) -> bool:
    cx1 = (o1_x1 + o1_x2) / 2
    cx2 = (o2_x1 + o2_x2) / 2
    return cx1 > cx2

def left_of(o1_x1: float, o1_x2: float, o2_x1: float, o2_x2: float) -> bool:
    cx1 = (o1_x1 + o1_x2) / 2
    cx2 = (o2_x1 + o2_x2) / 2
    return cx1 < cx2

def front_of(o1_y1: float, o1_y2: float, o2_y1: float, o2_y2: float) -> bool:
    cy1 = (o1_y1 + o1_y2) / 2
    cy2 = (o2_y1 + o2_y2) / 2
    return cy1 > cy2

def behind(o1_y1: float, o1_y2: float, o2_y1: float, o2_y2: float) -> bool:
    cy1 = (o1_y1 + o1_y2) / 2
    cy2 = (o2_y1 + o2_y2) / 2
    return cy1 < cy2

def register_step_interpreters(dataset='nlvr', use_precomputed=False, module_list=None):
    if dataset=='nlvr':
        return dict(
            VQA=VQAInterpreter(),
            EVAL=EvalInterpreter(),
            RESULT=ResultInterpreter()
        )
    elif dataset=='gqa':
        return dict(
            LOC=LocInterpreter(),
            COUNT=CountInterpreter(),
            CROP=CropInterpreter(),
            CROP_RIGHTOF=CropRightOfInterpreter(),
            CROP_LEFTOF=CropLeftOfInterpreter(),
            CROP_FRONTOF=CropFrontOfInterpreter(),
            CROP_INFRONTOF=CropInFrontOfInterpreter(),
            CROP_INFRONT=CropInFrontInterpreter(),
            CROP_BEHIND=CropBehindInterpreter(),
            CROP_AHEAD=CropAheadInterpreter(),
            CROP_BELOW=CropBelowInterpreter(),
            CROP_ABOVE=CropAboveInterpreter(),
            VQA=VQAInterpreter(),
            EVAL=EvalInterpreter(),
            RESULT=ResultInterpreter()
        )
    elif dataset=='gqa_video':
        return dict(
            LOC=LocVideoInterpreter(),
            COUNT=CountVideoInterpreter(),
            CROP=CropVideoInterpreter(),
            CROP_RIGHTOF=CropRightOfVideoInterpreter(),
            CROP_LEFTOF=CropLeftOfVideoInterpreter(),
            CROP_FRONTOF=CropFrontOfVideoInterpreter(),
            CROP_INFRONTOF=CropInFrontOfVideoInterpreter(),
            CROP_INFRONT=CropInFrontVideoInterpreter(),
            CROP_BEHIND=CropBehindVideoInterpreter(),
            CROP_AHEAD=CropAheadVideoInterpreter(),
            CROP_BELOW=CropBelowVideoInterpreter(),
            CROP_ABOVE=CropAboveVideoInterpreter(),
            EVAL=EvalVideoInterpreter(),
            RESULT=ResultVideoInterpreter(),
            BEFORE=BeforeVideoInterpreter(),
            AFTER=AfterVideoInterpreter(),
        )
    elif dataset=='clevr':
        all_modules = dict(
            LOC=LocClevrInterpreter(use_precomputed),
            BIG=BigClevrInterpreter(),
            SMALL=SmallClevrInterpreter(),
            GRAY=GrayClevrInterpreter(),
            RED=RedClevrInterpreter(),
            BLUE=BlueClevrInterpreter(),
            GREEN=GreenClevrInterpreter(),
            BROWN=BrownClevrInterpreter(),
            PURPLE=PurpleClevrInterpreter(),
            CYAN=CyanClevrInterpreter(),
            YELLOW=YellowClevrInterpreter(),
            CUBE=CubeClevrInterpreter(),
            SPHERE=SphereClevrInterpreter(),
            CYLINDER=CylinderClevrInterpreter(),
            RUBBER=RubberClevrInterpreter(),
            METAL=MetalClevrInterpreter(),
            LEFTOF=LeftOfClevrInterpreter(),
            RIGHTOF=RightOfClevrInterpreter(),
            FRONTOF=FrontOfClevrInterpreter(),
            BEHIND=BehindClevrInterpreter(),
            EQUALSIZE=EqualSizeClevrInterpreter(),
            EQUALMATERIAL=EqualMaterialClevrInterpreter(),
            EQUALSHAPE=EqualShapeClevrInterpreter(),
            EQUALCOLOR=EqualColorClevrInterpreter(),
            EVAL=EvalInterpreter(),
            RESULT=ResultInterpreter()
        )
        if module_list is None:
            return all_modules
        else:
            registered_modules = {key: all_modules[key] for key in module_list}
            return registered_modules
    elif dataset=='clevrer':
        all_modules = dict(
            LOC=LocClevrerInterpreter(use_precomputed),
            TRACK=TrackClevrerInterpreter(use_precomputed),
            LEFT=LeftClevrerInterpreter(),
            RIGHT=RightClevrerInterpreter(),
            TOP=TopClevrerInterpreter(),
            BOTTOM=BottomClevrerInterpreter(),
            GRAY=GrayClevrerInterpreter(),
            RED=RedClevrerInterpreter(),
            BLUE=BlueClevrerInterpreter(),
            GREEN=GreenClevrerInterpreter(),
            BROWN=BrownClevrerInterpreter(),
            PURPLE=PurpleClevrerInterpreter(),
            CYAN=CyanClevrerInterpreter(),
            YELLOW=YellowClevrerInterpreter(),
            CUBE=CubeClevrerInterpreter(),
            SPHERE=SphereClevrerInterpreter(),
            CYLINDER=CylinderClevrerInterpreter(),
            RUBBER=RubberClevrerInterpreter(),
            METAL=MetalClevrerInterpreter(),
            LEFTOF=LeftOfClevrerInterpreter(),
            RIGHTOF=RightOfClevrerInterpreter(),
            FRONTOF=FrontOfClevrerInterpreter(),
            BEHIND=BehindClevrerInterpreter(),
            EVENT=EventClevrerInterpreter(),
            BEFORE=BeforeClevrerInterpreter(),
            EVAL=EvalInterpreter(),
            RESULT=ResultInterpreter()
            # TODO: Add NEAR and FAR?
        )
        duckdb.create_function("Left", left)
        duckdb.create_function("Right", right)
        duckdb.create_function("Top", top)
        duckdb.create_function("Bottom", bottom)
        duckdb.create_function("LeftOf", left_of)
        duckdb.create_function("RightOf", right_of)
        duckdb.create_function("FrontOf", front_of)
        duckdb.create_function("Behind", behind)
        if module_list is None:
            return all_modules
        else:
            registered_modules = {key: all_modules[key] for key in module_list}
            return registered_modules
    elif dataset=='imageEdit':
        return dict(
            FACEDET=FaceDetInterpreter(),
            SEG=SegmentInterpreter(),
            SELECT=SelectInterpreter(),
            COLORPOP=ColorpopInterpreter(),
            BGBLUR=BgBlurInterpreter(),
            REPLACE=ReplaceInterpreter(),
            EMOJI=EmojiInterpreter(),
            RESULT=ResultInterpreter()
        )
    elif dataset=='okDet':
        return dict(
            FACEDET=FaceDetInterpreter(),
            LIST=ListInterpreter(),
            CLASSIFY=ClassifyInterpreter(),
            RESULT=ResultInterpreter(),
            TAG=TagInterpreter(),
            LOC=Loc2Interpreter(thresh=0.05,nms_thresh=0.3)
        )