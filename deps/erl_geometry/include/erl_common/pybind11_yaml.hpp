#pragma once

#include "erl_common/pybind11.hpp"
#include "erl_common/yaml.hpp"

inline void
BindYamlableBase(const py::module &m) {
    using namespace erl::common;
    py::class_<YamlableBase, std::shared_ptr<YamlableBase>>(m, "YamlableBase", py::module_local())
        .def("as_yaml_string", &YamlableBase::AsYamlString)
        .def("as_yaml_file", &YamlableBase::AsYamlFile, py::arg("yaml_file"))
        .def("from_yaml_string", &YamlableBase::FromYamlString, py::arg("yaml_str"))
        .def("from_yaml_file", &YamlableBase::FromYamlFile, py::arg("yaml_file"))
        .def("__str__", [](const YamlableBase &self) -> std::string {
            std::stringstream ss;
            ss << self;
            return ss.str();
        });
}
