import argparse
import logging
import yaml
import os
from PIL import Image
from tqdm import tqdm
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_ccot', action='store_true', help='use ccot technique')
    parser.add_argument('--save_output', action='store_true', help='save the output')
    parser.add_argument('--output_dir', type=str, help='where to save the results')
    parser.add_argument('--run_id', type=int, help='run id')
    parser.add_argument('--question_id', type=int, help='question id')
    parser.add_argument('--task_name', type=str, help='task name')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    args = parser.parse_args()
    use_ccot = args.use_ccot
    save_output = args.save_output
    output_dir = args.output_dir
    run_id = args.run_id
    question_id = args.question_id
    task_name = args.task_name
    batch_size = args.batch_size

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))

    num_images = 15000

    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_path = "/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_4bit=True,
        # bnb_4bit_compute_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        # device_map="auto",
    # )
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"

    system_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    # read json file
    with open(os.path.join(config['data_dir'], "clevr", "{}.json".format(task_name)), "r") as f:
        data = json.load(f)
    question = data['questions'][question_id]['question']
    gt_positive_images = data['questions'][question_id]['positive_images']
    if use_ccot:
        s_in = """For the provided image and its associated question, generate a scene graph in JSON format that includes the following: 1. Objects that are relevant to answering the question. 2. Object attributes that are relevant to answering the question. 3. Object relationships that are relevant to answering the question. Scene Graph:
        """
        p_in = f"Question: \"Is the following description correct for this image? {question}\""
        stage1_prompt = f"{system_prompt} USER: <image>\n{p_in}\n{s_in}\nASSISTANT:"
        context = "Use the image and scene graph as context and answer the following question:"
        print("stage1_prompt:", stage1_prompt)
    else:
        prompt = f"{system_prompt} USER: <image>\nIs the following description correct for this image? {question} Answer with 'yes' or 'no'.\nASSISTANT:"
        print(prompt)

    gt_labels = []
    for fid in range(num_images):
        if fid in gt_positive_images:
            gt_labels.append(1)
        else:
            gt_labels.append(0)

    pred_positive_images = []
    img_dir = os.path.join(config['data_dir'], "clevr/images/test")

    for i in tqdm(range(0, num_images, batch_size)):
        images = []
        for j in range(batch_size):
            if i + j >= num_images:
                break
            filename = f"CLEVR_test_{str(i + j).zfill(6)}.png"
            filepath = os.path.join(img_dir, filename)
            image = Image.open(filepath)
            image = expand2square(image, tuple(int(x*255) for x in processor.image_processor.image_mean))
            images.append(image)
        if use_ccot:
            # Stage 1
            stage1_prompts = [stage1_prompt] * len(images)
            inputs = processor(text=stage1_prompts, images=images, padding=True, return_tensors="pt").to(device, torch.float16)
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.2,
                top_p=0.7,
            )
            generated_text = processor.batch_decode(output, skip_special_tokens=True)
            # Stage 2
            stage2_prompts = []
            for j, text in enumerate(generated_text):
                result = text.split("ASSISTANT:")[-1].strip()
                # print(f"image {i+j} scene graph: {result}")
                s_g = f"Scene Graph:\n{result}"
                stage2_prompt = f"{system_prompt} USER: <image>\n{s_g}\n{context}\n{p_in}\nAnswer with 'yes' or 'no'.\nASSISTANT:"
                stage2_prompts.append(stage2_prompt)
                print("stage2_prompt:", stage2_prompt)
            inputs = processor(text=stage2_prompts, images=images, padding=True, return_tensors="pt").to(device, torch.float16)
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.2,
                top_p=0.7,
            )
            generated_text = processor.batch_decode(output, skip_special_tokens=True)
        else:
            prompts = [prompt] * len(images)
            inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(device, torch.float16)
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.2,
                top_p=0.7,
            )
            generated_text = processor.batch_decode(output, skip_special_tokens=True)
        for j, text in enumerate(generated_text):
            result = text.split("ASSISTANT:")[-1].strip().lower()
            print(f"image {i+j}: {result}")
            if "yes" in result:
                pred_positive_images.append(i+j)
    print(pred_positive_images)

    pred_labels = []
    for fid in range(num_images):
        if fid in pred_positive_images:
            pred_labels.append(1)
        else:
            pred_labels.append(0)

    # Compute accuracy, F1, precision, recall
    accuracy = accuracy_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    precision = precision_score(gt_labels, pred_labels)
    recall = recall_score(gt_labels, pred_labels)
    print(f"Accuracy: {accuracy}")
    print(f"F1: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    if save_output:
        # Save the output
        output = {
            "question": question,
            "gt_positive_images": gt_positive_images,
            "pred_positive_images": pred_positive_images,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
        # create output directory if not exists
        dir_name = "with_ccot" if use_ccot else "without_ccot"
        if not os.path.exists(os.path.join(output_dir, 'llava_clevr', dir_name)):
            os.makedirs(os.path.join(output_dir, 'llava_clevr', dir_name))

        with open(os.path.join(output_dir, 'llava_clevr', dir_name, f"task_{task_name}_run_{run_id}_question_{question_id}.json"), "w") as f:
            json.dump(output, f)
