# origin: https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/Evaluation/HumanEval/human_eval/execution.py#L165

import contextlib
import signal
import os
import subprocess

gpp_exec = ""


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def cpp_compile_and_run(cpp_code, temp_dir, test_file="test.cpp", timeout=60, use_ssl=False):
    os.makedirs(temp_dir, exist_ok=True)
    open(os.path.join(temp_dir, f"{test_file}"), 'w').write(cpp_code)
    origin_path = os.getcwd()
    os.chdir(temp_dir)
    res = "failed: unknown error"
    compile_returncode = -1
    try:
        ssl_flag = ""
        if use_ssl:
            ssl_flag = "-lcrypto -lssl"
        cmd = f"{gpp_exec}g++ -std=c++17 {test_file} {ssl_flag}"
        compilation_result = subprocess.run(cmd, timeout=60, capture_output=True, shell=True)
        compile_returncode = compilation_result.returncode
        print(compilation_result)
    except subprocess.TimeoutExpired as e:
        pass

    if compile_returncode != 0:
        res = "failed: compilation error"
    else:
        try:
            with time_limit(timeout):
                # WARNING
                # This program exists to execute untrusted model-generated code. Although
                # it is highly unlikely that model-generated code will do something overtly
                # malicious in response to this test suite, model-generated code may act
                # destructively due to a lack of model capability or alignment.
                # Users are strongly encouraged to sandbox this evaluation suite so that it
                # does not perform destructive actions on their host or network.
                # Once you have read this disclaimer and taken appropriate precautions,
                # uncomment the following line and proceed at your own risk:
                exec_result = subprocess.run(["./a.out"], timeout=timeout, capture_output=True)
                if exec_result.returncode == 0:
                    res = "passed"
                elif exec_result.returncode == 1:
                    if "AssertionError" in exec_result.stderr.decode('unicode-escape'):
                        res = "failed: wrong answer"
                    else:
                        res = f"failed: {exec_result.stderr.decode()}"
        except TimeoutError as e:
            res = "failed: time out"
        except subprocess.TimeoutExpired as e:
            res = "failed: time out"
        except BaseException as e:
            res = f"failed: {e}"

    os.chdir(origin_path)
    return res

