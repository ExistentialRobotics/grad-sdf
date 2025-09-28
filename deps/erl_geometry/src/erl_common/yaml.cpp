#include "erl_common/yaml.hpp"

#ifdef ERL_USE_BOOST
    #include <boost/program_options.hpp>
#endif

#include <any>
#include <fstream>
#include <stack>
#include <unordered_map>

namespace erl::common {

    bool
    YamlableBase::operator==(const YamlableBase &other) const {
        const std::string yaml_str = AsYamlString();
        const std::string other_yaml_str = other.AsYamlString();
        // if the YAML string is the same, the object is the same in terms of saving and loading
        return yaml_str == other_yaml_str;
    }

    bool
    YamlableBase::operator!=(const YamlableBase &other) const {
        return !(*this == other);
    }

    bool
    YamlableBase::FromYamlString(const std::string &yaml_string) {
        const YAML::Node node = YAML::Load(yaml_string);
        return FromYamlNode(node);
    }

    std::string
    YamlableBase::AsYamlString() const {
        YAML::Emitter emitter;
        emitter.SetIndent(4);
        emitter.SetSeqFormat(YAML::Flow);
        emitter << AsYamlNode();
        return emitter.c_str();
    }

    bool
    YamlableBase::FromYamlFile(const std::string &yaml_file) {
        if (!std::filesystem::exists(yaml_file)) {
            ERL_WARN("File does not exist: {}", yaml_file);
            return false;
        }
        const auto node = YAML::LoadFile(yaml_file);
        return FromYamlNode(node);
    }

    void
    YamlableBase::AsYamlFile(const std::string &yaml_file) const {
        std::ofstream ofs(yaml_file);
        if (!ofs.good()) {
            ERL_WARN("Failed to open file: {}", yaml_file);
            return;
        }
        YAML::Emitter emitter(ofs);
        emitter.SetIndent(4);
        emitter.SetSeqFormat(YAML::Flow);
        emitter << AsYamlNode();
    }

    bool
    YamlableBase::Write(std::ostream &s) const {
        if (!s.good()) { return false; }
        const std::string yaml_str = AsYamlString() + "\n";
        const auto len = static_cast<std::streamsize>(yaml_str.size());
        s.write(reinterpret_cast<const char *>(&len), sizeof(len));
        s.write(yaml_str.data(), len);
        return s.good();
    }

    bool
    YamlableBase::Read(std::istream &s) {
        if (!s.good()) { return false; }
        std::streamsize len;
        s.read(reinterpret_cast<char *>(&len), sizeof(len));
        std::string yaml_str(len, '\0');
        s.read(yaml_str.data(), len);
        return FromYamlString(yaml_str) && s.good();
    }

    void
    YamlableBase::FromCommandLine(int argc, const char *argv[]) {
#ifdef ERL_USE_BOOST
        namespace po = boost::program_options;
        po::options_description desc;

        YAML::Node node = AsYamlNode();
        auto options = desc.add_options();
        std::unordered_map<std::string, std::pair<YAML::Node, std::vector<std::string>>> option_map;

        options("help,h", "Show help message");

        std::stack<std::pair<std::string, YAML::Node>> node_stack;
        node_stack.emplace("", node);
        while (!node_stack.empty()) {
            auto [prefix, current_node] = node_stack.top();
            node_stack.pop();

            if (current_node.IsMap()) {
                for (const auto &item: current_node) {
                    node_stack.emplace(
                        prefix.empty() ? item.first.as<std::string>()
                                       : prefix + "." + item.first.as<std::string>(),
                        item.second);
                }
                continue;
            }

            if (current_node.IsSequence()) {
                auto itr = option_map.try_emplace(
                    prefix,
                    std::pair{current_node, std::vector<std::string>()});
                ERL_ASSERTM(itr.second, "Failed to insert option: {}", prefix);
                options(
                    itr.first->first.c_str(),
                    po::value<std::vector<std::string>>(&itr.first->second.second)->multitoken(),
                    ("List of values for " + prefix).c_str());
                continue;
            }

            auto itr =
                option_map.try_emplace(prefix, std::pair{current_node, std::vector{std::string()}});
            options(
                itr.first->first.c_str(),
                po::value<std::string>(itr.first->second.second.data()),
                ("Value for " + prefix).c_str());
        }

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
        if (vm.count("help")) {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl << desc << std::endl;
            exit(0);
        }
        po::notify(vm);
        for (auto &[name, value]: option_map) {
            if (vm.count(name)) {
                if (auto &node_value = value.first; node_value.IsSequence()) {
                    node_value = value.second;
                } else {
                    node_value = value.second[0];
                }
            }
        }
        ERL_ASSERTM(FromYamlNode(node), "Failed to parse command line options. ");
#else
        (void) argc;
        (void) argv;
        ERL_FATAL("Not compiled with Boost. Cannot parse command line arguments.");
#endif
    }
}  // namespace erl::common
