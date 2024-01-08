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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_precomputed', action='store_true', help='use precomputed object detection results')
    args = parser.parse_args()
    use_precomputed = args.use_precomputed

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    interpreter = ProgramInterpreter(dataset='clevr', use_precomputed=use_precomputed)

    prompter = partial(create_prompt,method='random', num_prompts=18)
    generator = ProgramGenerator(prompter=prompter)

    # question = "a big object o1 is right of the brown cylinder o2 and left of the large brown sphere o3"
    question = "a big blue block"
    # question = "An cube has the same material as the gray object"
    prog,_ = generator.generate(dict(question=question))
    print(prog)

    outputs = []
    img_dir = "/home/enhao/VOCAL-UDF/data/clevr/images/test"
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
        result, prog_state = interpreter.execute(prog,init_state,inspect=False)
        if result == "yes":
            outputs.append(fid)
    print(outputs)