import importlib
import os
import pathlib
import shutil
import site
import subprocess
import sys

if sys.version_info.major == 3 and sys.version_info.minor < 11:
    import toml as tomllib
else:
    import tomllib

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

torch_dir = os.environ.get("Torch_DIR", None)
try:
    if torch_dir is None:
        import torch

        torch_dir = pathlib.Path(torch.__file__).parent / "share" / "cmake" / "Torch"
    print(f"Torch_DIR: {torch_dir}")
except ImportError:
    print("torch is not installed, the system libtorch will be used if needed.")
    pass

# read project configuration from pyproject.toml
with open("pyproject.toml", "r") as f:
    config = tomllib.loads("".join(f.readlines()))
python_pkg_name = config["erl"]["python_pkg_name"]
pybind_module_name = config["erl"]["pybind_module_name"]
cmake_build_type = config["erl"].get("build_type", "Release")
cmake_ignore_conda_libraries = config["erl"].get("ignore_conda_libraries", "ON")
cmake_use_lapack = config["erl"].get("use_lapack", "ON")
cmake_use_lapack_strict = config["erl"].get("use_lapack_strict", "OFF")
cmake_use_intel_mkl = config["erl"].get("use_intel_mkl", "ON")
cmake_use_aocl = config["erl"].get("use_aocl", "OFF")
cmake_use_single_threaded_blas = config["erl"].get("use_single_threaded_blas", "ON")
cmake_use_tracy = config["erl"].get("use_tracy", "OFF")
cmake_use_pangolin = config["erl"].get("use_pangolin", "OFF")
cmake_use_plplot = config["erl"].get("use_plplot", "ON")
cmake_use_libtorch = config["erl"].get("use_libtorch", "OFF")
cmake_build_test = config["erl"].get("build_test", "OFF")
force_cmake_reconfigure = config["erl"].get("force_cmake_reconfigure", "ON")

erl_dependencies = config["erl"]["erl_dependencies"]

# detect if ROS1 is enabled
if os.environ.get("ROS_VERSION", "0") == "1":
    catkin = importlib.import_module("catkin_pkg.python_setup")
    setup_args = catkin.generate_distutils_setup(
        packages=find_packages("python"),
        package_dir={python_pkg_name: f"python/{python_pkg_name}"},
    )
    setup(**setup_args)
    exit(0)

# ROS not enabled, build the package from source
# check if cmake is installed
cmake_paths = [
    "/usr/bin/cmake",
    "/usr/local/bin/cmake",
]
cmake_path = None
for path in cmake_paths:
    if os.path.exists(path):
        cmake_path = path
        break
cmake_path = os.environ.get("CMAKE_EXECUTABLE", cmake_path)
assert cmake_path is not None, f"cmake is not found in {cmake_paths}"

# load configuration from environment variables
cmake_build_type = os.environ.get("BUILD_TYPE", cmake_build_type)
cmake_ignore_conda_libraries = os.environ.get("IGNORE_CONDA_LIBRARIES", cmake_ignore_conda_libraries)
cmake_use_lapack = os.environ.get("USE_LAPACK", cmake_use_lapack)
cmake_use_lapack_strict = os.environ.get("USE_LAPACK_STRICT", cmake_use_lapack_strict)
cmake_use_intel_mkl = os.environ.get("USE_INTEL_MKL", cmake_use_intel_mkl)
cmake_use_aocl = os.environ.get("USE_AOCL", cmake_use_aocl)
cmake_use_single_threaded_blas = os.environ.get("USE_SINGLE_THREADED_BLAS", cmake_use_single_threaded_blas)
cmake_use_tracy = os.environ.get("USE_TRACY", cmake_use_tracy)
cmake_use_pangolin = os.environ.get("USE_PANGOLIN", cmake_use_pangolin)
cmake_use_plplot = os.environ.get("USE_PLPLOT", cmake_use_plplot)
cmake_use_libtorch = os.environ.get("USE_LIBTORCH", cmake_use_libtorch)
cmake_build_test = os.environ.get("BUILD_TEST", cmake_build_test)
force_cmake_reconfigure = os.environ.get("FORCE_CMAKE_RECONFIGURE", force_cmake_reconfigure)

available_build_types = ["Release", "Debug", "RelWithDebInfo"]
assert cmake_build_type in available_build_types, f"build type {cmake_build_type} is not in {available_build_types}"
clean_before_build = os.environ.get("CLEAN_BEFORE_BUILD", "0") == "1"
n_proc = os.environ.get("N_JOBS", os.cpu_count())

# compute paths
project_dir = pathlib.Path(__file__).resolve().parent  # the directory of setup.py, should be the project root
src_python_dir = project_dir / "python" / python_pkg_name  # the directory of python source code
egg_info_dir = project_dir / f"{python_pkg_name}.egg-info"  # the directory of egg-info
build_dir = project_dir / "build" / cmake_build_type  # the build directory

# print configuration
print("====================================================================================================")
print("Configuration:")
print("----------------------------------------------------------------------------------------------------")
print(f"Project directory: {project_dir}")
print(f"Python source directory: {src_python_dir}")
print(f"Egg-info directory: {egg_info_dir}")
print(f"Build directory: {build_dir}")
print(f"Clean before build: {clean_before_build}")
print(f"Number of threads: {n_proc}")
print(f"cmake executable: {cmake_path}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"CMAKE_BUILD_TYPE: {cmake_build_type}")
print(f"ERL_IGNORE_CONDA_LIBRARIES: {cmake_ignore_conda_libraries}")
print(f"ERL_USE_LAPACK: {cmake_use_lapack}")
print(f"ERL_USE_LAPACK_STRICT: {cmake_use_lapack_strict}")
print(f"ERL_USE_INTEL_MKL: {cmake_use_intel_mkl}")
print(f"ERL_USE_AOCL: {cmake_use_aocl}")
print(f"ERL_USE_SINGLE_THREADED_BLAS: {cmake_use_single_threaded_blas}")
print(f"ERL_USE_TRACY: {cmake_use_tracy}")
print(f"ERL_USE_PANGOLIN: {cmake_use_pangolin}")
print(f"ERL_USE_PLPLOT: {cmake_use_plplot}")
print(f"ERL_USE_LIBTORCH: {cmake_use_libtorch}")
print(f"ERL_BUILD_TEST: {cmake_build_test}")
print(f"FORCE_CMAKE_RECONFIGURE: {force_cmake_reconfigure}")
print("====================================================================================================")

# clean up
if egg_info_dir.exists():
    os.system(f"rm -rf {egg_info_dir}")
if clean_before_build:
    os.system(f"rm -rf {build_dir}")


# build the package
class CMakeExtension(Extension):
    def __init__(self, name: str, source_dir: str = project_dir):
        super().__init__(name, sources=[])
        self.source_dir = os.path.abspath(source_dir)


class CMakeBuild(build_ext):
    def run(self) -> None:
        try:
            subprocess.check_output([cmake_path, "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext: CMakeExtension) -> None:
        original_full_path: str = self.get_ext_fullpath(ext.name)
        if os.path.exists(original_full_path):
            os.remove(original_full_path)

        # ext_dir equals to {build_dir}/lib.linux-$(architecture)-cpython-${python_version}
        editable = os.path.dirname(original_full_path) == project_dir  # editable install
        if editable:
            ext_dir = src_python_dir
        else:
            ext_dir = pathlib.Path(original_full_path).resolve().parent / self.distribution.get_name()
        old_ext_path = ext_dir / os.path.basename(original_full_path)
        if old_ext_path.exists():
            os.remove(old_ext_path)
        build_temp = build_dir / ext.name
        if build_temp.exists() and clean_before_build:
            shutil.rmtree(build_temp)
        os.makedirs(build_temp, exist_ok=True)
        os.makedirs(ext_dir, exist_ok=True)
        if force_cmake_reconfigure.upper() == "ON" and (build_temp / "CMakeCache.txt").exists():
            os.remove(build_temp / "CMakeCache.txt")
        if not (build_temp / "CMakeCache.txt").exists():
            cmake_args = [
                f"-DPython3_ROOT_DIR:PATH={pathlib.Path(sys.executable).parent.parent}",
                f"-DCMAKE_BUILD_TYPE={cmake_build_type}",
                f"-DCMAKE_INSTALL_PREFIX:PATH={ext_dir}",  # used to install the package
                f"-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON",
                f"-DERL_IGNORE_CONDA_LIBRARIES:BOOL={cmake_ignore_conda_libraries}",
                f"-DERL_USE_LAPACK:BOOL={cmake_use_lapack}",
                f"-DERL_USE_LAPACK_STRICT:BOOL={cmake_use_lapack_strict}",
                f"-DERL_USE_INTEL_MKL:BOOL={cmake_use_intel_mkl}",
                f"-DERL_USE_AOCL:BOOL={cmake_use_aocl}",
                f"-DERL_USE_SINGLE_THREADED_BLAS:BOOL={cmake_use_single_threaded_blas}",
                f"-DERL_USE_TRACY:BOOL={cmake_use_tracy}",
                f"-DERL_USE_PANGOLIN:BOOL={cmake_use_pangolin}",
                f"-DERL_USE_PLPLOT:BOOL={cmake_use_plplot}",
                f"-DERL_USE_LIBTORCH:BOOL={cmake_use_libtorch}",
                f"-DERL_BUILD_TEST:BOOL={cmake_build_test}",
                f"-DPIP_LIB_DIR:PATH={ext_dir}",
            ]
            if torch_dir is not None:
                cmake_args.append(f"-DTorch_DIR:PATH={torch_dir}")
            # add dependencies
            site_packages_dir = site.getsitepackages()[0]
            user_site_packages_dir = site.getusersitepackages()
            for dependency in erl_dependencies:
                cmake_dirs = [
                    f"{site_packages_dir}/{dependency}/share/{dependency}/cmake",
                    f"{user_site_packages_dir}/{dependency}/share/{dependency}/cmake",
                ]
                cmake_dir = None
                for cmake_dir in cmake_dirs:
                    if os.path.exists(cmake_dir):
                        break
                if cmake_dir is None:
                    raise RuntimeError(f"dependency {dependency} is not installed")
                cmake_args.append(f"-D{dependency}_DIR:PATH={cmake_dir}")
            # run cmake configure
            subprocess.check_call([cmake_path, ext.source_dir] + cmake_args, cwd=build_temp)
        # run cmake build and install
        subprocess.check_call(
            [cmake_path, "--build", ".", "--target", "install", "--", "-j", f"{n_proc}"],
            cwd=build_temp,
        )


if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r") as f:
        requires = f.readlines()
else:
    requires = []
for i, require in enumerate(requires):
    if require.startswith("git"):
        left, pkg_name = require.split("=")
        pkg_name = pkg_name.strip()
        requires[i] = f"{pkg_name} @ {require.strip()}"

setup(
    name=python_pkg_name,
    description=config["erl"]["description"],
    version=config["erl"]["version"],
    author=config["erl"]["author"],
    author_email=config["erl"]["author_email"],
    license=config["erl"]["license"],
    ext_modules=[CMakeExtension(pybind_module_name)],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=requires,
    packages=find_packages("python"),
    package_dir={python_pkg_name: f"python/{python_pkg_name}"},
    include_package_data=True,
    entry_points=config["erl"]["entry_points"],
)
