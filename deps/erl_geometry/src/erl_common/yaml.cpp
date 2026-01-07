#include "erl_common/yaml.hpp"

#include <any>
#include <fstream>
#include <stack>
#include <string>

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

    static void
    FromYamlFileRecursive( // NOLINT(*-no-recursion)
        const std::string &yaml_file,
        const std::string &base_field,
        YAML::Node &node) {
        if (!std::filesystem::exists(yaml_file)) {
            ERL_WARN("File does not exist: {}", yaml_file);
            return;
        }

        auto cur_node = YAML::LoadFile(yaml_file);
        const auto yaml_dir = std::filesystem::absolute(yaml_file).parent_path();

        if (!base_field.empty()) {
            // first process all base files in child nodes
            for (auto item: cur_node) {
                if (!item.second.IsMap()) { continue; }
                if (!item.second[base_field].IsDefined()) { continue; }
                const auto base_file_str = item.second[base_field].as<std::string>();
                if (base_file_str.empty()) { continue; }
                auto base_file = std::filesystem::path(base_file_str);
                if (!base_file.is_absolute()) { base_file = yaml_dir / base_file; }
                YAML::Node child_node(YAML::NodeType::Map);
                FromYamlFileRecursive(base_file, base_field, child_node);
                UpdateYamlNode(item.second, child_node, UnknownFieldPolicy::kMerge);
                item.second = child_node;
            }

            // then process the base file of the current node
            std::string base_file_str;
            if (cur_node[base_field].IsDefined()) {
                base_file_str = cur_node[base_field].as<std::string>();
            }

            if (!base_file_str.empty()) {
                auto base_file = std::filesystem::path(base_file_str);
                if (!base_file.is_absolute()) { base_file = yaml_dir / base_file; }
                FromYamlFileRecursive(base_file, base_field, node);
                UpdateYamlNode(cur_node, node, UnknownFieldPolicy::kMerge);
                return;
            }
        }

        // top-level file or no base file specified
        if (node.size() == 0) {
            node = cur_node;  // assign it directly, more efficient
            return;
        }
        UpdateYamlNode(cur_node, node, UnknownFieldPolicy::kMerge);
    }

    bool
    YamlableBase::FromYamlFile(const std::string &yaml_file, const std::string &base_config_field) {
        if (!std::filesystem::exists(yaml_file)) {
            ERL_WARN("File does not exist: {}", yaml_file);
            return false;
        }
        YAML::Node node(YAML::NodeType::Map);
        FromYamlFileRecursive(yaml_file, base_config_field, node);
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
        std::streamsize len = 0;
        s.read(reinterpret_cast<char *>(&len), sizeof(len));
        std::string yaml_str(len, '\0');
        s.read(yaml_str.data(), len);
        return FromYamlString(yaml_str) && s.good();
    }

    bool
    YamlableBase::FromCommandLine(const std::vector<std::string> &args) {
        std::vector<const char *> argv_vec;
        argv_vec.reserve(args.size());
        for (const auto &arg: args) { argv_vec.push_back(arg.c_str()); }
        return FromCommandLine(static_cast<int>(args.size()), argv_vec.data());
    }

    bool
    YamlableBase::FromCommandLine(const int argc, char *argv[]) {
        std::vector<const char *> argv_vec;
        argv_vec.reserve(argc);
        for (int i = 0; i < argc; ++i) { argv_vec.push_back(argv[i]); }
        return FromCommandLine(argc, argv_vec.data());
    }

    bool
    YamlableBase::FromCommandLine(const int argc, const char *argv[]) {
#ifdef ERL_USE_BOOST
        program_options::ProgramOptionsData po_data(
            fmt::format("Options for {}", type_name(*this)),
            120);

        namespace po = boost::program_options;

        std::string config_file;
        po_data.desc.add_options()           //
            ("help,h", "Show help message")  //
            ("config",
             po::value<std::string>(&config_file)->value_name("CONFIG"),
             "Path to YAML config file");
        auto parsed =
            po::command_line_parser(argc, argv).options(po_data.desc).allow_unregistered().run();
        po::store(parsed, po_data.vm);
        po::notify(po_data.vm);
        po_data.print_help = po_data.vm.count("help") > 0;

        if (!config_file.empty()) {
            ERL_ASSERTM(
                po_data.print_help || std::filesystem::exists(config_file),
                "Config file does not exist: {}",
                config_file);
            ERL_ASSERTM(
                po_data.print_help || FromYamlFile(config_file),
                "Failed to load config file: {}",
                config_file);
        }

        po_data.args = po::collect_unrecognized(parsed.options, po::include_positional);
        po_data.vm.clear();

        FromCommandLineImpl(po_data, "");
        if (po_data.print_help) {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                      << po_data.desc << std::endl;
            exit(EXIT_SUCCESS);
        }

        bool success = true;

        if (!po_data.args.empty()) {
            ERL_ERROR("Unrecognized arguments: {}", po_data.args);
            success = false;
        }

        if (!po_data.Successful()) {
            ERL_WARN("Failed to parse command line arguments, errors:\n" + po_data.error_msgs);
            success = false;
        }

        if (!success) { return false; }

        return this->PostDeserialization();  // call post deserialization hook
#else
        (void) argc;
        (void) argv;
        ERL_ERROR("Not compiled with Boost. Cannot parse command line arguments.");
        return false;
#endif
    }

    void
    UpdateYamlNode(
        const YAML::Node &src,
        YAML::Node &dst,
        const UnknownFieldPolicy unknown_field_policy) {

        std::stack<std::pair<YAML::Node, YAML::Node>> node_stack;
        node_stack.emplace(src, dst);
        while (!node_stack.empty()) {
            auto [current_src, current_dst] = node_stack.top();
            node_stack.pop();

            ERL_ASSERTM(current_src.IsMap(), "Source node must be a map.");
            ERL_ASSERTM(current_dst.IsMap(), "Destination node must be a map.");

            for (const auto &item: current_src) {
                const auto &key = item.first.as<std::string>();
                const auto &src_value = item.second;

                if (!current_dst[key].IsDefined()) {
                    switch (unknown_field_policy) {
                        case UnknownFieldPolicy::kIgnore:
                            break;
                        case UnknownFieldPolicy::kMerge:
                            current_dst[key] = YAML::Clone(src_value);
                            break;
                        case UnknownFieldPolicy::kWarn:
                            ERL_WARN("Unknown field {} in the dst node");
                            break;
                        case UnknownFieldPolicy::kError:
                            ERL_FATAL("Unknown field {} in the dst node");
                    }
                    continue;
                }

                if (src_value.IsMap()) {
                    const YAML::Node dst_value = current_dst[key];
                    ERL_ASSERTM(
                        dst_value.IsMap(),
                        "Destination node must be a map for key: {}",
                        key);
                    node_stack.emplace(src_value, dst_value);
                } else {
                    current_dst[key] = YAML::Clone(src_value);
                }
            }
        }
    }

}  // namespace erl::common
