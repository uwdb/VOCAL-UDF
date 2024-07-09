import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from PIL import Image
from functools import partial

from vocaludf.utils import StreamToLogger, exception_hook
from visprog.engine.utils import ProgramGenerator, ProgramInterpreter
from visprog.prompts.clevrer import create_prompt
import argparse
import logging
from tqdm import tqdm
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import yaml
import cv2
import random


logger = logging.getLogger("vocaludf")
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_precomputed', action='store_true', help='use precomputed object detection results')
    parser.add_argument("--num_missing_udfs", type=int, help="number of missing UDFs")
    parser.add_argument('--save_output', action='store_true', help='save the output')
    parser.add_argument('--output_dir', type=str, help='where to save the results')
    parser.add_argument('--run_id', type=int, help='run id')
    parser.add_argument('--question_id', type=int, help='question id')
    parser.add_argument('--task_name', type=str, help='task name')
    parser.add_argument('--llm_model', type=str, default="gpt-3.5-turbo-instruct", help='llm model', choices=['gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-1106', 'gpt-4-turbo-2024-04-09'])
    args = parser.parse_args()
    use_precomputed = args.use_precomputed
    save_output = args.save_output
    output_dir = args.output_dir
    run_id = args.run_id
    question_id = args.question_id
    task_name = args.task_name
    llm_model = args.llm_model

    config = yaml.safe_load(open("/gscratch/balazinska/enhaoz/VOCAL-UDF/configs/config.yaml", "r"))

    """
    Set up logging
    """
    # Create a directory if it doesn't already exist
    log_dir = os.path.join(
        config["log_dir"],
        "visprog",
        'clevrer',
        llm_model,
    )
    os.makedirs(log_dir, exist_ok=True)

    # Create a file handler that logs even debug messages
    file_handler = logging.FileHandler(os.path.join(log_dir, f"task_{task_name}_run_{run_id}_question_{question_id}.log"), mode="w")
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler with a higher log level
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)

    # logger.addHandler(console_handler)
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    sys.excepthook = exception_hook

    # read json file
    with open(os.path.join(config['data_dir'], "clevrer", "3_new_udfs_labels.json"), "r") as f:
        data = json.load(f)
    question = data['questions'][question_id]['question']
    gt_positive_videos = data['questions'][question_id]['positive_videos']
    new_modules = data['questions'][question_id]['new_modules']
    logger.info(f"question: {question}")
    logger.info(f"new_modules: {new_modules}")

    # Base modules
    module_list = ["LOC", "TRACK", "GRAY", "RED", "BLUE", "GREEN", "CUBE", "SPHERE", "RUBBER", "LEFTOF", "FRONTOF", "LEFT", "TOP", "EVENT", "BEFORE", "EVAL", "RESULT"]
    random.seed(run_id)
    if "0" in task_name: # VisProg has access to all UDFs
        module_list = module_list + new_modules
    elif "1" in task_name: # VisProg doesn't have access to 1 UDF
        module_list = module_list + random.sample(new_modules, 2)
    elif "2" in task_name: # VisProg doesn't have access to 2 UDFs
        module_list = module_list + random.sample(new_modules, 1)
    elif "3" in task_name: # VisProg doesn't have access to 3 UDFs
        module_list = module_list
    logger.info(f"module_list: {module_list}")
    interpreter = ProgramInterpreter(dataset='clevrer', use_precomputed=use_precomputed, module_list=module_list)

    prompt_modules = '\n'.join(module_list)
    prompt_modules = f'You can only use modules below to generate the program:\n{prompt_modules}'

    prompter = partial(create_prompt, method='random', num_prompts=8, seed=run_id, prompt_modules=prompt_modules)
    generator = ProgramGenerator(prompter=prompter, temperature=config['visprog']['program_generator']['temperature'],top_p=config['visprog']['program_generator']['top_p'], llm_model=llm_model)

    gt_labels = []
    for vid in range(10000):
        if vid in gt_positive_videos:
            gt_labels.append(1)
        else:
            gt_labels.append(0)

    for retry in range(5):
        prog,_ = generator.generate(dict(question=question),retry)
        logger.info(f"prog: {prog}")

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
            logger.info(f"pred_positive_videos: {pred_positive_videos}")
            break
        except Exception as e:
            logger.exception(e)
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

    logger.info(output)