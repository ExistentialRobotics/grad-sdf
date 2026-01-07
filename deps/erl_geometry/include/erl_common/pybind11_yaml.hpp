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
        .def(
            "from_yaml_file",
            &YamlableBase::FromYamlFile,
            py::arg("yaml_file"),
            py::arg("base_config_field"))
        .def(
            "from_command_line",
            py::overload_cast<const std::vector<std::string> &>(&YamlableBase::FromCommandLine),
            py::arg("args"))
        .def("__str__", [](const YamlableBase &self) -> std::string {
            std::stringstream ss;
            ss << self;
            return ss.str();
        });
}

template<typename M, typename T, typename B = erl::common::YamlableBase>
auto
BindYamlable(const M &m, const char *name) {
    using namespace erl::common;
    py::class_<T, B, std::shared_ptr<T>> cls(m, name);
    cls.def(py::init());
    std::apply(
        [&](const auto &...member_info) {
            (cls.def_readwrite(member_info.name, member_info.ptr), ...);
        },
        T::Schema);
    return cls;
}

template<typename M, typename T, int N>
auto
BindYamlableEnum(const M &m, const char *name) {
    py::enum_<T> cls(m, name, py::arithmetic());
    std::apply(
        [&](const auto &...member_info) { (cls.value(member_info.name, member_info.value), ...); },
        MakeEnumSchema<T, N>());
    cls.export_values();
    return cls;
}
