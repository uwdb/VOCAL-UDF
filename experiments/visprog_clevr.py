import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from PIL import Image
from functools import partial

from visprog.engine.utils import ProgramGenerator, ProgramInterpreter
from visprog.prompts.clevr import create_prompt
import argparse
import logging
from tqdm import tqdm
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_precomputed', action='store_true', help='use precomputed object detection results')
    parser.add_argument('--save_output', action='store_true', help='save the output')
    parser.add_argument('--output_dir', type=str, help='where to save the results')
    parser.add_argument('--run_id', type=int, help='run id')
    parser.add_argument('--question_id', type=int, help='question id')
    parser.add_argument('--task_name', type=str, help='task name')
    args = parser.parse_args()
    use_precomputed = args.use_precomputed
    save_output = args.save_output
    output_dir = args.output_dir
    run_id = args.run_id
    question_id = args.question_id
    task_name = args.task_name

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))

    module_list = ["LOC", "BIG", "GRAY", "RED", "BLUE", "GREEN", "CUBE", "SPHERE", "RUBBER", "LEFTOF", "FRONTOF", "EQUALSIZE", "EQUALMATERIAL", "EVAL", "RESULT"]

    interpreter = ProgramInterpreter(dataset='clevr', use_precomputed=use_precomputed, module_list=module_list)

    prompt_modules = '\n'.join(module_list)
    prompt_modules = f'You can only use modules below:\n{prompt_modules}'

    prompter = partial(create_prompt, method='random', num_prompts=18, seed=run_id, prompt_modules=prompt_modules)
    print(prompter)
    generator = ProgramGenerator(prompter=prompter)

    # read json file

    with open(os.path.join(config['data_dir'], "clevr", "{}.json".format(task_name)), "r") as f:
        data = json.load(f)
    question = data['questions'][question_id]['question']
    gt_positive_images = data['questions'][question_id]['positive_images']
    print(question)

    gt_labels = []
    for fid in range(15000):
        if fid in gt_positive_images:
            gt_labels.append(1)
        else:
            gt_labels.append(0)

    prog,_ = generator.generate(dict(question=question))
    print(prog)

    pred_positive_images = []
    img_dir = os.path.join(config['data_dir'], "clevr/images/test")
    failed = 0
    for fid in tqdm(range(15000)):
        filename = f"CLEVR_test_{str(fid).zfill(6)}.png"
        filepath = os.path.join(img_dir, filename)
        image = Image.open(filepath)
        image.thumbnail((640,640),Image.Resampling.LANCZOS)
        init_state = dict(
            IMAGE=image.convert('RGB')
        )
        if use_precomputed:
            init_state['fid'] = fid
        try:
            result, prog_state = interpreter.execute(prog,init_state,inspect=False)
        except:
            failed += 1
            continue
        if result == "yes":
            pred_positive_images.append(fid)
    print(pred_positive_images)

    pred_labels = []
    for fid in range(15000):
        if fid in pred_positive_images:
            pred_labels.append(1)
        else:
            pred_labels.append(0)

    # Compute accuracy, F1, precision, recall
    accuracy = accuracy_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    precision = precision_score(gt_labels, pred_labels)
    recall = recall_score(gt_labels, pred_labels)
    print("Accuracy: ", accuracy)
    print("F1: ", f1)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("# Failures: ", failed)
    if save_output:
        # Save the output
        output = {
            "question": question,
            "prog": prog,
            "gt_positive_images": gt_positive_images,
            "pred_positive_images": pred_positive_images,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "failed": failed
        }
        with open(os.path.join(output_dir, f"task_{task_name}_run_{run_id}_question_{question_id}.json"), "w") as f:
            json.dump(output, f)



