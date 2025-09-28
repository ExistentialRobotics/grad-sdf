import glob

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

_ext_sources = glob.glob("src/*.cpp")

setup(
    name="sparse_octree",
    ext_modules=[
        CppExtension(
            name="sparse_octree.svo",
            sources=_ext_sources,
            extra_compile_args={"cxx": ["-g", "-O2", "-DNDEBUG"]},
        )
    ],
    packages=["sparse_octree"],
    cmdclass={"build_ext": BuildExtension},
)
