from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.eval.run_llava import eval_model
import requests
from sklearn.metrics import f1_score

import argparse
import os
import json
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import math
import matplotlib.pyplot as plt

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def llava_test_example_image():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # model_id = "llava-hf/llava-1.5-13b-hf"
    model_id = "/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/llava-1.5-7b-hf"
    # model_id = "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/llava-v1.6-vicuna-7b"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        # low_cpu_mem_usage=True,
        # load_in_4bit=True,
        # attn_implementation="flash_attention_2",
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    # processor.tokenizer.padding_side = "left"

    # model.save_pretrained("/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/llava-1.5-13b-hf")
    # processor.save_pretrained("/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/llava-1.5-13b-hf")

    # prompt = "<image>\nUSER: What's the content of the image? \nASSISTANT:"
    # prompt = "<image>\nUSER: what are the colors of the objects?\nASSISTANT:"
    query =  "A yellow object o1 is to the right of and of equal size to an object o2, which is to the left of and of equal shape to a rubber object o3; Object o3 is in front of o2."

    prompt = f"USER: <image>\nIs the following description correct for this image? {query}\nASSISTANT:"

    fid = 0
    image = Image.open(f"/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevr/images/test/CLEVR_test_{str(fid).zfill(6)}.png")

    inputs = processor(prompt, image, return_tensors="pt").to(device, torch.float16)
    # inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(device, torch.float16)

    # Generate
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.2,
        top_p=0.7,
    )
    print(processor.decode(output[0][2:], skip_special_tokens=True))
    # generated_text = processor.batch_decode(output, skip_special_tokens=True)
    # for text in generated_text:
        # print(text.split("ASSISTANT:")[-1])

    # print(processor.decode(output[0][2:], skip_special_tokens=True))
    # generate_ids = model.generate(**inputs, max_length=30)
    # processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def llava_naive_batch():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # model_id = "llava-hf/llava-1.5-13b-hf"
    model_id = "/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/llava-1.5-7b-hf"
    # model_id = "/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/llava-v1.6-vicuna-7b"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_4bit=True,
        attn_implementation="flash_attention_2",
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"

    # model.save_pretrained("/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/llava-1.5-13b-hf")
    # processor.save_pretrained("/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/llava-1.5-13b-hf")

    query =  "A yellow object o1 is to the right of and of equal size to an object o2, which is to the left of and of equal shape to a rubber object o3; Object o3 is in front of o2."

    prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nIs the following description correct for this image? {query} Answer with 'yes' or 'no'.\nASSISTANT:"

    batch_size = 8
    images = []
    fid = 0
    prompts = [prompt] * batch_size
    for fid in range(batch_size):
        image = Image.open(f"/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevr/images/test/CLEVR_test_{str(fid).zfill(6)}.png")
        # IMPORTANT: pad the image to square
        image = expand2square(image, tuple(int(x*255) for x in processor.image_processor.image_mean))
        images.append(image)

    inputs = processor(prompt, image, return_tensors="pt").to(device, torch.float16)
    # inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(device, torch.float16)

    # Generate
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.2,
        top_p=0.7,
    )
    print(processor.decode(output[0][2:], skip_special_tokens=True))

def llava_naive_serial():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # model_id = "llava-hf/llava-1.5-13b-hf"
    model_id = "/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    query =  "A yellow object o1 is to the right of and of equal size to an object o2, which is to the left of and of equal shape to a rubber object o3; Object o3 is in front of o2."

    prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nIs the following description correct for this image? {query} Answer with 'yes' or 'no'.\nASSISTANT:"

    num_images = 15000
    preds = []
    for fid in tqdm(range(num_images)):
        image = Image.open(f"/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevr/images/test/CLEVR_test_{str(fid).zfill(6)}.png")
        # IMPORTANT: pad the image to square
        image = expand2square(image, tuple(int(x*255) for x in processor.image_processor.image_mean))
        inputs = processor(prompt, image, return_tensors="pt").to(device, torch.float16)
        # Generate
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.2,
            top_p=0.7,
        )

        output = processor.decode(output[0][2:], skip_special_tokens=True)
        print(output)
        if "yes" in output.lower():
            preds.append(1)
        else:
            preds.append(0)
    print("preds", preds)
    with open("/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevr/3_new_udfs_labels.json", "r") as f:
        data = json.load(f)
    gt_pos = data['questions'][0]['positive_images']
    gt_labels = []
    for i in range(num_images):
        if i in gt_pos:
            gt_labels.append(1)
        else:
            gt_labels.append(0)
    f1 = f1_score(gt_labels, preds)
    print("f1:", f1)

def llava_ccot_batch():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    # model_id = "llava-hf/llava-1.5-7b-hf"
    model_id = "/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/llava-1.5-7b-hf"

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_4bit=True,
        # bnb_4bit_compute_dtype=torch.float16,
        # device_map="auto",
        attn_implementation="flash_attention_2",
    # )
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"

    query =  "A yellow object o1 is to the right of and of equal size to an object o2, which is to the left of and of equal shape to a rubber object o3; Object o3 is in front of o2."

    s_in = """For the provided image and its associated question, generate a scene graph in JSON format that includes the following: 1. Objects that are relevant to answering the question. 2. Object attributes that are relevant to answering the question. 3. Object relationships that are relevant to answering the question. Scene Graph:
    """
    p_in = f"Question: \"Is the following description correct for this image? {query}\""
    stage1_prompt = f"USER: <image>\n{p_in} {s_in}\nASSISTANT:"
    context = "Use the image and scene graph as context and answer the following question:"

    batch_size = 1
    for i in range(1):
        images = []
        for fid in range(batch_size):
            image = Image.open(f"/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevr/images/test/CLEVR_test_{str(fid+i*batch_size).zfill(6)}.png")
            image = Image.open(f"/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevr/images/test/CLEVR_test_000003.png")
            images.append(image)
        # Stage 1
        print(stage1_prompt)
        stage1_prompts = [stage1_prompt] * batch_size
        inputs = processor(text=stage1_prompts, images=images, padding=True, return_tensors="pt").to(device, torch.float16)
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.2,
            top_p=0.7,
        )
        generated_text = processor.batch_decode(output, skip_special_tokens=True)
        # Stage 2
        stage2_prompts = []
        for j, text in enumerate(generated_text):
            result = text.split("ASSISTANT:")[-1].strip()
            print(f"image {j} scene graph: {result}")
            s_g = f"Scene Graph: {result}"
            stage2_prompt = f"USER: <image>\n{s_g} {context} {p_in} Answer the question using a single word or phrase.\nASSISTANT:"
            stage2_prompts.append(stage2_prompt)
        inputs = processor(text=stage2_prompts, images=images, padding=True, return_tensors="pt").to(device, torch.float16)
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.2,
            top_p=0.7,
        )
        generated_text = processor.batch_decode(output, skip_special_tokens=True)
        for text in generated_text:
            print(text.split("ASSISTANT:")[-1])

def llava_github_naive(model_path):
    # Model
    disable_torch_init()
    # model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    num_images = 15000
    preds = []
    for fid in tqdm(range(num_images)):
        image_file = f"/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevr/images/test/CLEVR_test_{str(fid).zfill(6)}.png"
        query =  "A yellow object o1 is to the right of and of equal size to an object o2, which is to the left of and of equal shape to a rubber object o3; Object o3 is in front of o2."
        qs = f"Is the following description correct for this image? {query} Answer with 'yes' or 'no'."
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print("prompt", prompt)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image = Image.open(image_file).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True,
                temperature=0.2,
                top_p=0.7,
                num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=512,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(outputs)
        if "yes" in outputs.lower():
            preds.append(1)
        else:
            preds.append(0)
    print("preds", preds)
    with open("/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevr/3_new_udfs_labels.json", "r") as f:
        data = json.load(f)
    gt_pos = data['questions'][0]['positive_images']
    gt_labels = []
    for i in range(num_images):
        if i in gt_pos:
            gt_labels.append(1)
        else:
            gt_labels.append(0)
    f1 = f1_score(gt_labels, preds)
    print("f1:", f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    args = parser.parse_args()
    model_path = args.model_path
    # llava_naive_batch()
    # llava_test_example_image()
    # llava_github_naive(model_path)
    llava_naive_serial()
    # llava_ccot_batch()