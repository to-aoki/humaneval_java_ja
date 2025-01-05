# origin: https://github.com/deepseek-ai/DeepSeek-Coder/
# This code is a Japanese translation of the evaluation data with only Java evaluation sites extracted.

import os
import re
import json

from tqdm import tqdm
from openai import OpenAI

from java_runner import java_compile_and_run
from cpp_runner import cpp_compile_and_run


class HumanEvalTask:
    def __init__(self, name: str, language: str, prompt: str, doctests: str, original: str,
                 prompt_terminology: str, tests: str, stop_tokens: list, task_id: str, test: str):
        self.name = name
        self.language = language
        self.prompt = prompt
        self.doctests = doctests
        self.original = original
        self.prompt_terminology = prompt_terminology
        self.tests = tests
        self.stop_tokens = stop_tokens
        self.task_id = task_id
        self.test = test

    def to_dict(self):
        return {
            "name": self.name,
            "language": self.language,
            "prompt": self.prompt,
            "doctests": self.doctests,
            "original": self.original,
            "prompt_terminology": self.prompt_terminology,
            "tests": self.tests,
            "stop_tokens": self.stop_tokens,
            "task_id": self.task_id,
            "test": self.test
        }

    @classmethod
    def from_json(cls, file_path: str) -> list['HumanEvalTask']:
        tasks = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                task = HumanEvalTask(
                    name=data['name'],
                    language=data['language'],
                    prompt=data['prompt'],
                    doctests=data['doctests'],
                    original=data['original'],
                    prompt_terminology=data['prompt_terminology'],
                    tests=data['tests'],
                    stop_tokens=data['stop_tokens'],
                    task_id=data['task_id'],
                    test=data['test']
                )
                tasks.append(task)
        return tasks

    @classmethod
    def save_json(cls, tasks: list['HumanEvalTask'], file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            for task in tasks:
                f.write(json.dumps(task.to_dict()) + "\n")

    _hira_kana_pattern = re.compile(r'[ぁ-ゖァ-ヺー]')

    @classmethod
    def contains_hira_kana(cls, text: str) -> bool:
        # 翻訳漏れチェック用
        lines = text.splitlines()
        for line in lines:
            if cls._hira_kana_pattern.search(line):
                return True
        return False

    def get_function_name(self):
        func_lines = [x for x in self.prompt.strip().split('\n') if x.strip()]

        if self.language.lower() == 'python':
            func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("def ")][-1]
            func_name = func_lines[func_idx].split('(')[0].strip()
            func_prefix = "\n".join(func_lines[:func_idx])
            return func_name, func_prefix

        func_name = func_lines[-1].split('{')[0].strip()
        func_prefix = "\n".join(func_lines[:-1])
        return func_name, func_prefix

    def eval_code(self, generation: str) -> str:
        try:
            code_block: str = re.findall(f'```{self.language.lower()}\n(.*?)```', generation, re.DOTALL | re.IGNORECASE)[0]

            # Remove main and last "}"
            if self.language == 'java':
                main_func = "public static void main"
                last_count = 2
            elif self.language == 'cpp':
                main_func = "int main"
                last_count = 1
            escaped_signature = re.escape(main_func)
            pattern = rf'{escaped_signature}\s*\([^)]*\)\s*\{{(?:[^{{}}]*|\{{[^{{}}]*\}})*\}}'
            code_block = re.sub(pattern, '', code_block, flags=re.DOTALL)

            code_block = code_block[::-1].replace('}', '', last_count)[::-1]
            lines = code_block.splitlines()
            cleaned_lines = [line for line in lines if line.strip()]
            return "\n".join(cleaned_lines) + '\n' + self.test

        except Exception as e:
            print(e)
            return "failed: parse error"


client = OpenAI(
    base_url='http://127.0.0.1:8080/',
    api_key=os.environ.get("OPENAI_API_KEY", "empty"),
)

model = 'dummy'


def build_deepseekcoder_instruction_ja(question: str, languge='java'):
    if languge == 'java':
        return '''
関数を完成させてください。与えられたコードを修正することは許されません。完成した関数はコードブロックにまとめて返してください。与えられたコードを含めて返却してください。以下は完成させるためのコードです：
```{}
{}
```
'''.strip().format(languge.lower(), question.strip())
    elif languge == 'cpp':
        return '''
関数を完成させてください。与えられたコードを修正することは許されません。ただし不足するinclude文は追加してください。完成した関数はコードブロックにまとめて返してください。与えられたコードを含めて返却してください。以下は完成させるためのコードです：
```{}
{}
```
'''.strip().format(languge.lower(), question.strip())
    else:
        raise ValueError('not supported languge: {}'.format(languge))


def code_complete(incomplete_codes: list[str], languge='java'):
    results = []
    for incomplete_code in tqdm(incomplete_codes, desc='generating'):
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": build_deepseekcoder_instruction_ja(incomplete_code, languge)
                }
            ],
            model=model,
        )
        results.append(
            response.choices[0].message.content
        )
    return results


def evaluate(file_path='data/humaneval-java_ja.jsonl', temp_path='./tmp', sampling=-1):
    tasks = HumanEvalTask.from_json(file_path)[:sampling]
    prompts = [t.prompt for t in tasks]
    language = tasks[0].language
    generations = code_complete(prompts, language)
    build_targets = [t.eval_code(g) for [t, g] in zip(tasks, generations)]
    results = []
    for [t, b] in tqdm(zip(tasks, build_targets), desc='compile and run'):
        if b.startswith('failed'):
            results.append(b)
            continue
        if language == 'java':
            results.append(
                java_compile_and_run(java_code=b, temp_dir=os.path.join(temp_path, t.name)))
        elif language == 'cpp':
            # 不足するincludeを動的に補う（元コード踏襲。ここで評価しないなら最初からつけていいかも）
            cpp_includes = [
                "#include<stdlib.h>",
                "#include<algorithm>",
                "#include<math.h>",
                "#include<stdio.h>",
                "#include<vector>",
                "#include<string>",
                "#include<climits>",
                "#include<cstring>",
                "#include<iostream>",
                "#include<cassert>"
            ]
            test_set_up = ''
            for s in cpp_includes:
                if s not in b:
                    test_set_up += s + "\n"
            test_string = test_set_up + "\n" + b
            use_ssl = False
            if t.task_id == 162:
                # コンパイル都合
                use_ssl = True
            results.append(
                cpp_compile_and_run(cpp_code=test_string, temp_dir=os.path.join(temp_path, t.name), use_ssl=use_ssl))
    return results, tasks


def main():
    import argparse
    parser = argparse.ArgumentParser(description="HumanEval java or cpp (ja)")
    parser.add_argument('--file_path', type=str, default='data/humaneval-cpp_ja.jsonl')
    parser.add_argument('--temp_dir', type=str, default='./tmp')
    parser.add_argument('--sampling', type=int, default=-1)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    results, tasks = evaluate(file_path=args.file_path, temp_path=args.temp_dir, sampling=args.sampling)
    passed = 0
    for r in results:
        if r == 'passed':
            passed += 1
    print('pass@1:', str(passed / len(results)))
    if args.show:
        for [t, r] in zip(tasks, results):
            print(t.name, r)


if __name__ == "__main__":
    main()
