#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
import re
import subprocess
from sysconfig import get_path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir: str = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # if not os.path.exists(extdir):
        #     os.mkdir(extdir)

        cfg = "Debug" if self.debug else "Release"

        try:
            import torch
            torch_cmake_path = torch.utils.cmake_prefix_path
        except Exception:
            print("PyTorch not installed.")
            exit(-1)

        py_version = ''.join(re.match(r"\d+.\d+", sys.version).group(0).split('.'))

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(extdir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DPYTHON_INCLUDE_DIR={}".format(get_path('include')),  # not used on MSVC, but no harm
            "-DTORCH_CMAKE_PATH={}".format(torch_cmake_path),
            "-DPYTHON_VERSION={}".format(py_version)
        ]
        print(cmake_args)
        build_args = []

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=self.build_temp)

# Setup GESCPP
setup(
      name="gescpp",
      version="0.0.1",
      description="GES in cpp",
      ext_modules=[CMakeExtension("gescpp")],
      cmdclass={"build_ext": CMakeBuild},
      zip_safe=False
)
