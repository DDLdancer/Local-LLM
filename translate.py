#!/usr/bin/env python

import argparse
from llama_cpp import Llama
from tqdm import tqdm

ENG_MAX_LENGTH = 4096
JP_MAX_LENGTH = 1024
MAX_TOKENS = 0  # 0 means depends on n_ctx
N_CTX = 4096  # 0 means using the model's default context length
ENABLE_FLASH_ATTN = False
N_GPU_LAYERS = -1  # -1 means offloading all layers to GPU

ENG_DELIMITERS = ['\n', '.', '?', '!']
JP_DELIMITERS = ['\n', '！', '？', '。', '、']
MODEL_PATH = "/home/youchengzhang/llm/model-qwen2.5-32b-instruct-q4_k_m/qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf"
INPUT_PATH = "input.txt"
OUTPUT_PATH = "output.txt"
STOP = "翻译完成"

parser = argparse.ArgumentParser(description="Translate text from English to Chinese.")
parser.add_argument("-v", "--verbose", action="store_true", help="Output translated text while processing.")
parser.add_argument("-j", "--japanese", action="store_true", help="target text is Japanese")
args = parser.parse_args()

# 初始化LLM模型
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=N_GPU_LAYERS,
    flash_attn=ENABLE_FLASH_ATTN,
    n_ctx=N_CTX,
    verbose=args.verbose,
)


def split_text_with_priority_delimiters(text, max_length, delimiters):
    parts = []
    while text:
        if len(text) <= max_length:
            parts.append((text, ''))
            break
        # 查找基于优先级的最合适的分割点
        best_position = -1
        delimiter_used = ''
        for delim in delimiters:
            position = text.rfind(delim, 0, max_length)
            if position != -1:
                best_position = position
                delimiter_used = delim
                break  # 找到最高优先级的标点后停止搜索

        if best_position == -1:  # 如果没有找到任何标点符号，使用最大长度
            best_position = max_length - 1
            parts.append((text[:best_position + 1], ''))
        else:
            best_position += 1  # 包括分隔符本身
            parts.append((text[:best_position], delimiter_used))
        text = text[best_position:]
    return parts


def translate_text(text, max_length, delimiters, verbose):
    # 将文本分割为适合模型的大小
    parts = split_text_with_priority_delimiters(text, max_length, delimiters)
    translated_parts = []

    # 逐一翻译每部分
    for part, delim in tqdm(parts):
        prompt = f"Q: 翻译如下文本到中文，翻译完成后输出“{STOP}”: {part} A: 以下是文本的中文翻译: "
        result = llm(prompt, max_tokens=MAX_TOKENS, stop=STOP)
        translated_text = result["choices"][0]["text"].strip()
        if delim == '\n':
            translated_text += '\n'  # 如果使用的分隔符是换行符，则在翻译文本后添加换行符
        if verbose:
            print(part)
            print(translated_text)
        translated_parts.append(translated_text)

    # 返回拼接后的翻译文本
    return "".join(translated_parts)


# 读取文本文件
with open(INPUT_PATH, 'r', encoding='utf-8') as file:
    content = file.read()

# 翻译文本
if args.japanese:
    translated_content = translate_text(content, JP_MAX_LENGTH, JP_DELIMITERS, args.verbose)
else:
    translated_content = translate_text(content, ENG_MAX_LENGTH, ENG_DELIMITERS, args.verbose)


# 保存翻译结果
with open(OUTPUT_PATH, 'w', encoding='utf-8') as file:
    file.write(translated_content)
