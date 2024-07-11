import torch
from datasets import load_dataset
import json
from transformers import LlamaForCausalLM
from transformers import AutoTokenizer
from tqdm import tqdm
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
import time
import numpy as np
from transformers import LlamaConfig
import copy

from utils_hh.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
from utils_hh.modify_gptneox import convert_kvcache_gpt_neox_heavy_recent, GPTNeoXAttention_Mask
from utils_hh.modify_opt import convert_kvcache_opt_heavy_recent, OPTAttention_Mask

dataset_path = "/nfs/shared_LLM_dataset/LongBench/LongBench.py"
MODEL_PATH = "/nfs/shared_LLM_model/meta-llama/Llama-2-7b-chat-hf"
device = "cuda:0"
torch.cuda.set_device(device)

dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
model2maxlen = json.load(open("config/model2maxlen.json"))

ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": convert_kvcache_llama_heavy_recent,
    "opt": convert_kvcache_opt_heavy_recent,
    "gpt_neox": convert_kvcache_gpt_neox_heavy_recent,
}


def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


# datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
#             "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
#             "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
datasets = ["gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

model_name = "llama2-7b-chat-4k"

print("Begin Load Config and model and so on")

config = LlamaConfig.from_pretrained(MODEL_PATH)
config.heavy_ratio = 0.1
config.recent_ratio = 0.1

# replace_llama_attn_with_flash_attn()
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = LlamaForCausalLM.from_pretrained(MODEL_PATH,
                                         torch_dtype=torch.float16).to(device)
model = model.eval()
max_length = model2maxlen[model_name]

print("Begin")

checkpoint = copy.deepcopy(model.state_dict())
model = ENABLE_Heavy_Hitter_FUNCTIONS['llama'](model, config)
model.load_state_dict(checkpoint)

model = model.to(torch.float16).eval().to(device)
print(torch.cuda.max_memory_allocated())
print(torch.cuda.max_memory_reserved())
torch.cuda.empty_cache()
print("finish init")

# datasets = ["narrativeqa"]


def get_pred(device, data, max_length, max_gen, prompt_format, dataset,
             model_name, model, tokenizer, out_path):
    infer_time = []
    infer_count = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)

        start_time = time.time()
        tokenized_prompt = tokenizer(prompt,
                                     truncation=False,
                                     return_tensors="pt").input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(prompt,
                                         truncation=False,
                                         return_tensors="pt",
                                         add_special_tokens=False).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half],
                skip_special_tokens=True) + tokenizer.decode(
                    tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
                "trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            if dataset in [
                    "trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"
            ]:
                input = tokenizer(prompt,
                                  truncation=False,
                                  return_tensors="pt").to(device)
            else:
                input = prompt.to(device)
        else:
            input = tokenizer(prompt, truncation=False,
                              return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        if dataset == "samsum":  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1]
                ],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]
        pred = tokenizer.decode(output[context_length:],
                                skip_special_tokens=True)
        end_time = time.time()

        infer_time.append(end_time - start_time)
        infer_count.append(json_obj["length"])

        pred = post_process(pred, model_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"]
                },
                f,
                ensure_ascii=False)
            f.write('\n')

    print("infer_time:", np.sum(infer_time[5:]))
    print("infer_TPOT:", np.mean(infer_time[5:]) / np.mean(infer_count[5:]))


print(model.device)
root_path = "/root/workspace/pred"
framework_name = "h2o"
for dataset in datasets:
    data = load_dataset(dataset_path, dataset, split='test')
    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]
    out_path = f"{root_path}/{framework_name}/{dataset}.jsonl"
    begin = time.time()
    get_pred(device, data, max_length, max_gen, prompt_format, dataset,
             model_name, model, tokenizer, out_path)
    end = time.time()
    print("Max Allocated", torch.cuda.max_memory_allocated())
    print("Max Reserved", torch.cuda.max_memory_reserved())
    print(f"time: {end - begin}")
    torch.cuda.empty_cache()
