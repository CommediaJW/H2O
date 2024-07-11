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


class custom_dataset(torch.utils.data.Dataset):

    def __init__(self,
                 tokenizer,
                 data,
                 prompt,
                 max_length,
                 dataset,
                 model_name,
                 device="cuda"):
        super(custom_dataset, self).__init__()
        self.data = []
        self.tokenizer = tokenizer
        self.device = device
        for it, json_obj in enumerate(data):
            prompt = prompt_format.format(**json_obj)
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            tokenized_prompt = tokenizer(prompt,
                                         truncation=False,
                                         return_tensors="pt").input_ids[0]
            if "chatglm3" in model_name:
                tokenized_prompt = tokenizer(
                    prompt,
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
            self.data.append(prompt)
        self.length = len(self.data)

    def __getitem__(self, index):
        batch = self.data[index]
        return batch

    def __len__(self):
        return self.length


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
datasets = ["2wikimqa"]

model_name = "llama2-7b-chat-4k"

print("Begin Load Config and model and so on")

config = LlamaConfig.from_pretrained(MODEL_PATH)
config.heavy_ratio = 0.1
config.recent_ratio = 0.1

# replace_llama_attn_with_flash_attn()
tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"
model = LlamaForCausalLM.from_pretrained(MODEL_PATH,
                                         torch_dtype=torch.float16).to(device)
model = model.eval()
max_length = model2maxlen[model_name]

print("Begin")

checkpoint = copy.deepcopy(model.state_dict())
model = ENABLE_Heavy_Hitter_FUNCTIONS['llama'](model, config)
model.load_state_dict(checkpoint)
model = model.to(torch.float16).eval().to(device)
model.config.pad_token_id = tokenizer.pad_token_id

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
print(torch.cuda.max_memory_allocated())
print(torch.cuda.max_memory_reserved())
torch.cuda.empty_cache()
print("finish init")


def get_pred(device, data, dataloader, max_gen, model_name, model, tokenizer,
             out_path):
    infer_time = []
    infer_count = []
    result_list = []
    for batch in tqdm(dataloader):
        start_time = time.time()
        input = tokenizer(batch,
                          truncation=False,
                          padding=True,
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
        end_time = time.time()
        infer_time.append(end_time - start_time)

    print("infer_time:", np.sum(infer_time[5:]))


print(model.device)
root_path = "/root/workspace/pred"
framework_name = "h2o"
for dataset in datasets:
    data = load_dataset(dataset_path, dataset, split='test')
    prompt_format = dataset2prompt[dataset]
    max_gen = dataset2maxlen[dataset]
    prompt_dataset = custom_dataset(tokenizer, data, prompt_format, max_length,
                                    dataset, model_name)

    for batch_size in [1, 2, 4, 8]:
        dataloader = torch.utils.data.DataLoader(dataset=prompt_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False)
        out_path = f"{root_path}/{framework_name}/{dataset}_batchsize{batch_size}.jsonl"
        print(out_path)

        begin = time.time()
        get_pred(device, data, dataloader, max_gen, model_name, model,
                 tokenizer, out_path)
        end = time.time()
        print("Max Allocated", torch.cuda.max_memory_allocated())
        print("Max Reserved", torch.cuda.max_memory_reserved())
        print(f"time: {end - begin}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
