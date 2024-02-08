from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, Blip2ForConditionalGeneration
from tqdm import tqdm

def blip2_naive_batch():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_id = "/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/blip2-opt-2.7b"

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # attn_implementation="flash_attention_2",
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"

    # model.save_pretrained("/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/blip2-opt-2.7b")
    # processor.save_pretrained("/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/models/blip2-opt-2.7b")

    query =  "A rubber cube is in front of a big object, which in turn is in front of and to the left of another rubber object."

    prompt = f"Question: Is the following description correct for this image? {query} Short answer:"

    batch_size = 32
    images = []
    prompts = [prompt] * batch_size
    for fid in range(batch_size):
        image = Image.open(f"/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevr/images/test/CLEVR_test_{str(fid).zfill(6)}.png")
        images.append(image)

    inputs = processor(text=prompts, images=images, padding=True, return_tensors="pt").to(device, torch.float16)

    # Generate
    output = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        num_beams=5,
        length_penalty=-1,
        # temperature=0.2,
        # top_p=0.7,
    )
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    for text in generated_text:
        print(text.split())

def blip2_ccot_batch():
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

    query =  "A rubber cube is in front of a big object, which in turn is in front of and to the left of another rubber object."

    s_in = """For the provided image and its associated question, generate a scene graph in JSON format that includes the following: 1. Objects that are relevant to answering the question. 2. Object attributes that are relevant to answering the question. 3. Object relationships that are relevant to answering the question. Scene Graph:
    """
    p_in = f"Question: \"Is the following description correct for this image? {query}\""
    stage1_prompt = f"USER: <image>\n{p_in} {s_in}\nASSISTANT:"
    context = "Use the image and scene graph as context and answer the following question:"

    batch_size = 16
    for i in range(4):
        images = []
        for fid in range(batch_size):
            image = Image.open(f"/mmfs1/gscratch/balazinska/enhaoz/VOCAL-UDF/data/clevr/images/test/CLEVR_test_{str(fid+i*batch_size).zfill(6)}.png")
            images.append(image)
        # Stage 1
        print(stage1_prompt)
        stage1_prompts = [stage1_prompt] * batch_size
        inputs = processor(text=stage1_prompts, images=images, padding=True, return_tensors="pt").to(device, torch.float16)
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            # temperature=0.2,
            # top_p=0.7,
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
            # temperature=0.2,
            # top_p=0.7,
        )
        generated_text = processor.batch_decode(output, skip_special_tokens=True)
        for text in generated_text:
            print(text.split("ASSISTANT:")[-1])

if __name__ == "__main__":
    blip2_naive_batch()
    # blip2_ccot_batch()