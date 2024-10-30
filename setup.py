from setuptools import setup, Extension
from torch.utils import cpp_extension
import torch
import setuptools
import os
import shutil
from distutils.cmd import Command

from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)
class CleanCommand(Command):
    description = "Clean up build and temporary files"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        here = os.path.abspath(os.path.dirname(__file__))
        build_dir = os.path.join(here, 'build')
        dist_dir = os.path.join(here, 'dist')
        egg_info_dir = os.path.join(here, 'dhellam.egg-info')
        lib = os.path.join(here, 'dhellam/_Clib.cpython-310-x86_64-linux-gnu.so')

        print("Cleaning up...")
        for d in [build_dir, dist_dir, egg_info_dir, lib]:
            if os.path.exists(d):
                print(f"Removing {d}")
                if os.path.isfile(d):
                    os.remove(d)
                else:
                    shutil.rmtree(d)
                

class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # free memory in GB
            max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)

with open("README.md", "r") as fh:
    long_description = fh.read()
def req_file(filename):
    with open(filename) as f:
        content = f.readlines()
    return [x.strip() for x in content]

install_requires = req_file("requirements.txt")


def compute_capability_args(cross_compile_archs=None):
    ccs = []
    for i in range(torch.cuda.device_count()):
        CC_MAJOR, CC_MINOR = torch.cuda.get_device_capability(i)
        cc = f"{CC_MAJOR}.{CC_MINOR}"
        if cc not in ccs:
            ccs.append(cc)
    ccs = sorted(ccs)
    ccs[-1] += '+PTX'

    args = []
    for cc in ccs:
        num = cc[0] + cc[2]
        args.append(f'-gencode=arch=compute_{num},code=sm_{num}')
        if cc.endswith('+PTX'):
            args.append(f'-gencode=arch=compute_{num},code=compute_{num}')
    return args


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "8"
    return nvcc_extra_args + ["--threads", nvcc_threads]


def nvcc_args():
    args = ['-O3']
    args += [
        '-std=c++17',
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]
    args += compute_capability_args()
    return args

generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

def strip_empty_entries(args):
    '''
    Drop any empty strings from the list of compile and link flags
    '''
    return [x for x in args if len(x) > 0]

from pathlib import Path

def find_files(directory, extensions):
    matches = []
    path_obj = Path(directory)
    for file in path_obj.rglob('*'):
        if file.suffix in extensions:
            relative_path = str(file.relative_to(directory))
            matches.append(str(path_obj.joinpath(relative_path)))
    return matches

directory_path = 'csrc'  # 假设从当前目录开始搜索
extensions = ('.cpp', '.cu', '.cc', '.c')
cpp_and_cu_files = find_files(directory_path, extensions)
this_dir = os.path.dirname(os.path.abspath(__file__))

entry_points = {
    'console_scripts': [
    ]
}

__description__="DHeLlam"
__contact_names__="Anonymity"
__url__="https://github.com/DHeLlam-502/dhellam.git"
__keywords__="hpc llm"
__license__="MIT"
__package_name__="dhellam"
__version__="0.0.1"

flash_atten_include = [
    Path(this_dir) / "3rdparty/flash-attention" / "csrc" / "flash_attn",
    Path(this_dir) / "3rdparty/flash-attention" / "csrc" / "flash_attn" / "src",
    Path(this_dir) / "3rdparty/flash-attention" / "csrc" / "cutlass" / "include",
]
flash_atten_sources=[
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim32_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim32_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim64_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim96_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim96_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim160_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim160_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim192_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim192_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim224_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim224_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim256_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_hdim256_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim32_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim32_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim64_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim64_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim96_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim96_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim160_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim160_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim192_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim192_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim224_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim224_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim256_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_bwd_hdim256_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim32_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim64_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim96_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim96_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim160_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim160_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim192_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim192_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim224_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim224_bf16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim256_fp16_sm80.cu",
        Path(this_dir) / "3rdparty/flash-attention" / "csrc/flash_attn/src/flash_fwd_split_hdim256_bf16_sm80.cu",
    ]
            
cpp_and_cu_files+=flash_atten_sources
# cpp_and_cu_files+=smctrl_sources
cpp_and_cu_files = [str(file) for file in cpp_and_cu_files]

setup(
        name=__package_name__,
        version=__version__,
        description=__description__,
        long_description=long_description,
        long_description_content_type="text/markdown",
        url=__url__,
        author=__contact_names__,
        maintainer=__contact_names__,
        license=__license__,
        python_requires='>=3.6',
        packages=setuptools.find_packages(exclude=(
            "3rdparty",
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "dhellam.egg-info",
        )),
        entry_points=entry_points,
        install_requires=install_requires,
        ext_modules=[cpp_extension.CUDAExtension(
                    name='dhellam._Clib',
                    sources=cpp_and_cu_files,
                    include_dirs=cpp_extension.include_paths(cuda=True)+[Path(this_dir) / 'csrc/operators/']+flash_atten_include,
                    library_dirs=cpp_extension.library_paths(cuda=True)+['/usr/local/cuda/lib64'],
                    extra_compile_args={
                            'cxx':['-O3', '-std=c++17'] + generator_flag,
                            'nvcc':strip_empty_entries(nvcc_args())+ generator_flag
                        },
                    extra_link_args=['-lcuda'],
                    language='c++')],

        cmdclass={'build_ext': NinjaBuildExtension, 'clean': CleanCommand},
        package_data={'dhellam':['dhellam/_Clib/*.pyi']})