import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from PIL import Image
from functools import partial

from visprog.engine.utils import ProgramGenerator, ProgramInterpreter
from visprog.prompts.clevrer import create_prompt
import argparse
import logging
from tqdm import tqdm
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import yaml
import cv2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_precomputed', action='store_true', help='use precomputed object detection results')
    parser.add_argument('--save_output', action='store_true', help='save the output')
    parser.add_argument('--output_dir', type=str, help='where to save the results')
    parser.add_argument('--run_id', type=int, help='run id')
    parser.add_argument('--question_id', type=int, help='question id')
    parser.add_argument('--task_name', type=str, help='task name')
    parser.add_argument('--llm_model', type=str, default="gpt-3.5-turbo-instruct", help='llm model', choices=['gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview'])
    args = parser.parse_args()
    use_precomputed = args.use_precomputed
    save_output = args.save_output
    output_dir = args.output_dir
    run_id = args.run_id
    question_id = args.question_id
    task_name = args.task_name
    llm_model = args.llm_model

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))

    module_list = ["LOC", "TRACK", "GRAY", "RED", "BLUE", "GREEN", "CUBE", "SPHERE", "RUBBER", "LEFTOF", "FRONTOF", "LEFT", "TOP", "EVENT", "BEFORE", "EVAL", "RESULT"]

    interpreter = ProgramInterpreter(dataset='clevrer', use_precomputed=use_precomputed, module_list=module_list)

    prompt_modules = '\n'.join(module_list)
    prompt_modules = f'You can only use modules below to generate the program:\n{prompt_modules}'

    prompter = partial(create_prompt, method='random', num_prompts=8, seed=run_id, prompt_modules=prompt_modules)
    generator = ProgramGenerator(prompter=prompter, temperature=config['visprog']['program_generator']['temperature'],top_p=config['visprog']['program_generator']['top_p'], llm_model=llm_model)

    # read json file
    with open(os.path.join(config['data_dir'], "clevrer", "{}.json".format(task_name)), "r") as f:
        data = json.load(f)
    question = data['questions'][question_id]['question']
    gt_positive_videos = data['questions'][question_id]['positive_videos']
    print(question)

    gt_labels = []
    for vid in range(10000):
        if vid in gt_positive_videos:
            gt_labels.append(1)
        else:
            gt_labels.append(0)

    for retry in range(5):
        prog,_ = generator.generate(dict(question=question),retry)
        print(prog)

        pred_positive_videos = []
        failed = 0
        try:
            for vid in tqdm(range(10000)):
                video = []
                if not use_precomputed:
                    cap = cv2.VideoCapture(
                        os.path.join(
                            config['data_dir'],
                            'clevrer',
                            f'video_{str(vid//1000*1000).zfill(5)}-{str((vid//1000+1)*1000).zfill(5)}',
                            f"video_{str(vid).zfill(5)}.mp4"
                        )
                    )
                    fid = 0
                    while True:
                        ret, frame = cap.read()  # Read the next frame from the video
                        if not ret:
                            break  # Break the loop if there are no more frames

                        # Convert the frame from BGR (OpenCV format) to RGB (PIL format)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Convert the frame to a PIL Image and append to the list
                        image = Image.fromarray(frame)
                        image.thumbnail((640,640),Image.Resampling.LANCZOS)
                        video.append(image)
                        fid += 1
                init_state = dict(
                    VIDEO=video,
                    vid=vid
                )
                result, prog_state = interpreter.execute(prog,init_state,inspect=False)
                if result == 'yes':
                    pred_positive_videos.append(vid)
            print(pred_positive_videos)
            break
        except Exception as e:
            print(e)
            failed += 1

    pred_labels = []
    for vid in range(10000):
        if vid in pred_positive_videos:
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
            "gt_positive_videos": gt_positive_videos,
            "pred_positive_videos": pred_positive_videos,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "failed": failed
        }
        # create output directory if not exists
        if not os.path.exists(os.path.join(output_dir, llm_model)):
            os.makedirs(os.path.join(output_dir, llm_model))

        with open(os.path.join(output_dir, llm_model, f"task_{task_name}_run_{run_id}_question_{question_id}.json"), "w") as f:
            json.dump(output, f)