# origin: https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/Evaluation/HumanEval/human_eval/execution.py#L492

import os
import subprocess

java_exec = ""
JAR_CLASS_PATH = ['./javatuples-1.2.jar']


def java_compile_and_run(java_code, temp_dir, class_name="Problem", timeout=60, jar_class_paths=JAR_CLASS_PATH):
    jar_path_list = [os.path.abspath(jar_path) for jar_path in jar_class_paths]
    jar_path = ':'.join(jar_path_list)
    os.makedirs(temp_dir, exist_ok=True)
    open(os.path.join(temp_dir, f"{class_name}.java"), 'w').write(java_code)
    origin_path = os.getcwd()
    os.chdir(temp_dir)
    res = "failed: unknown error"
    compile_returncode = -1
    for _ in range(5):
        try:
            cmd = f"{java_exec}javac -cp {jar_path} {class_name}.java"
            compilation_result = subprocess.run(cmd, timeout=60, capture_output=True, shell=True)
            compile_returncode = compilation_result.returncode
            break
        except subprocess.TimeoutExpired as e:
            continue
    if compile_returncode != 0:
        res = "failed: compilation error"
    else:
        exec_result = None
        try:
            # WARNING
            # This program exists to execute untrusted model-generated code. Although
            # it is highly unlikely that model-generated code will do something overtly
            # malicious in response to this test suite, model-generated code may act
            # destructively due to a lack of model capability or alignment.
            # Users are strongly encouraged to sandbox this evaluation suite so that it
            # does not perform destructive actions on their host or network.
            # Once you have read this disclaimer and taken appropriate precautions,
            # uncomment the following line and proceed at your own risk:
            cmd = f"{java_exec}java -ea -cp .:{jar_path} {class_name}"
            exec_result = subprocess.run(cmd, timeout=timeout, capture_output=True, shell=True)
            if exec_result.returncode == 0:
                res = "passed"
            elif exec_result.returncode == 1:
                if "AssertionError" in exec_result.stderr.decode('unicode-escape'):
                    res = "failed: wrong answer"
                else:
                    res = f"failed: {exec_result.stderr.decode()}"
        except subprocess.TimeoutExpired as e:
            res = "failed: time out"
        except BaseException as e:
            res = f"failed: {e}"

    os.chdir(origin_path)
    return res

