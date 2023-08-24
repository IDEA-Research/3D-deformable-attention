# ------------------------------------------------------------------------
# DFA3D
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the IDEA License, Version 1.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from mmcv (https://github.com/open-mmlab/mmcv)
# Copyright (c) OpenMMLab. All rights reserved
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright 2018-2019 Open-MMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------ 

import glob
import os
import platform
import re
from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup

EXT_TYPE = ''
try:
    import torch
    if torch.__version__ == 'parrots':
        from parrots.utils.build_extension import BuildExtension
        EXT_TYPE = 'parrots'
    else:
        from torch.utils.cpp_extension import BuildExtension
        EXT_TYPE = 'pytorch'
    cmd_class = {'build_ext': BuildExtension}
except ModuleNotFoundError:
    cmd_class = {}
    print('Skip building ext ops due to the absence of torch.')


def choose_requirement(primary, secondary):
    """If some version of primary requirement installed, return primary, else
    return secondary."""
    try:
        name = re.split(r'[!<>=]', primary)[0]
        get_distribution(name)
    except DistributionNotFound:
        return secondary

    return str(primary)


def parse_requirements(fname='requirements/runtime.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


install_requires = parse_requirements()

try:
    # OpenCV installed via conda.
    import cv2  # NOQA: F401
    major, minor, *rest = cv2.__version__.split('.')
    if int(major) < 3:
        raise RuntimeError(
            f'OpenCV >=3 is required but {cv2.__version__} is installed')
except ImportError:
    # If first not installed install second package
    CHOOSE_INSTALL_REQUIRES = [('opencv-python-headless>=3',
                                'opencv-python>=3')]
    for main, secondary in CHOOSE_INSTALL_REQUIRES:
        install_requires.append(choose_requirement(main, secondary))


def get_extensions():
    extensions = []
    

    ext_name = 'dfa3D._ext'
    from torch.utils.cpp_extension import CppExtension, CUDAExtension

    try:
        import psutil
        num_cpu = len(psutil.Process().cpu_affinity())
        cpu_use = max(4, num_cpu - 1)
    except (ModuleNotFoundError, AttributeError):
        cpu_use = 4

    os.environ.setdefault('MAX_JOBS', str(cpu_use))
    define_macros = []

    extra_compile_args = {'cxx': []}

    if platform.system() != 'Windows':
        extra_compile_args['cxx'] = ['-std=c++14']

    include_dirs = []
    
    project_dir = 'dfa3D/ops/csrc/'
    define_macros += [('WITH_CUDA', None)]
    cuda_args = os.getenv('CUDA_ARGS')
    if cuda_args and len(cuda_args)>0:
        cuda_args = os.getenv('CUDA_ARGS')
        extra_compile_args['nvcc'] = []
        if cuda_args:
            for cuda_arg in cuda_args.split(","):
                extra_compile_args['nvcc'].append(cuda_arg)
    else:
        extra_compile_args['nvcc'] = []
    op_files = glob.glob('./dfa3D/ops/csrc/*.cpp') + \
        glob.glob('./dfa3D/ops/csrc/cpu/*.cpp') + \
        glob.glob('./dfa3D/ops/csrc/cuda/*.cu') + \
        glob.glob('./dfa3D/ops/csrc/cuda/*.cpp')
    extension = CUDAExtension
    include_dirs.append(os.path.abspath('./dfa3D/ops/csrc/common'))
    include_dirs.append(os.path.abspath('./dfa3D/ops/csrc/common/cuda'))
   
    if 'nvcc' in extra_compile_args and platform.system() != 'Windows':
        extra_compile_args['nvcc'] += ['-std=c++14']

    ext_ops = extension(
        name=ext_name,
        sources=op_files,
        include_dirs=include_dirs,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)
    extensions.append(ext_ops)
    return extensions


setup(
    name='dfa3D',
    description='3D Deformable Attention',
    packages=find_packages(),
    author='Hongyang Li',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=install_requires,
    ext_modules=get_extensions(),
    cmdclass=cmd_class,
    zip_safe=False)
