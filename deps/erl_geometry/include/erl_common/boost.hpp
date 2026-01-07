#pragma once

#ifdef ERL_USE_BOOST
    #include "eigen.hpp"
    #include "logging.hpp"
    #include "opencv.hpp"
    #include "string_utils.hpp"

    #include <boost/program_options.hpp>
    #include <yaml-cpp/yaml.h>

    #include <cctype>
    #include <memory>
    #include <string>
    #include <unordered_map>
    #include <vector>

namespace erl::common::program_options {
    // Functionality for parsing program options using boost::program_options

    namespace po = boost::program_options;

    [[nodiscard]] inline std::string
    GetBoostOptionName(const std::string &prefix, const std::string &name) {
        if (prefix.empty()) { return name; }
        return prefix + "." + name;
    }

    struct ProgramOptionsData;

    struct ParseOptionBase {
        std::string option_name;      // name of the option in command line
        std::string value_name;       // name of the value shown in help message
        std::string docs;             // documentation string
        ProgramOptionsData *po_data;  // pointer to the program options data
        void *member_ptr;             // pointer to the member variable to be set

        ParseOptionBase(
            std::string option_name_in,
            ProgramOptionsData *po_data_in,
            void *member_ptr_in)
            : option_name(std::move(option_name_in)),
              value_name(option_name),
              po_data(po_data_in),
              member_ptr(member_ptr_in) {

            for (auto &c: value_name) {
                if (c == '_' || c == '.') { continue; }
                c = static_cast<char>(std::toupper(c));
            }
        }

        ParseOptionBase(const ParseOptionBase &) = delete;
        ParseOptionBase &
        operator=(const ParseOptionBase &) = delete;
        ParseOptionBase(ParseOptionBase &&) = delete;
        ParseOptionBase &
        operator=(ParseOptionBase &&) = delete;

        virtual void
        Run() = 0;

        virtual ~ParseOptionBase() = default;  // enable polymorphism
    };

    template<typename M>
    struct ParseOption;

    struct ProgramOptionsData {
        std::vector<std::string> args;  // remaining args to be parsed
        po::options_description desc;   // description of program options
        po::variables_map vm;           // map of parsed variables
        bool print_help = false;        // whether to print help message
        std::string error_msgs;         // error messages

        // Map from option name to its parser.
        // This is used to keep the option parsers alive across multiple Parse() calls.
        // Some parsers need to store intermediate data (e.g., for Eigen matrices).
        // So we need to keep them in memory.
        std::unordered_map<std::string, std::shared_ptr<ParseOptionBase>> option_parsers;

        explicit ProgramOptionsData(const std::string &title, const unsigned line_length = 80)
            : desc(title, line_length) {}

        template<typename M>
        std::shared_ptr<ParseOption<M>>
        GetOptionParser(const std::string &option_name, M *member_ptr_in) {
            using T = ParseOption<M>;
            if (const auto it = option_parsers.find(option_name); it != option_parsers.end()) {
                auto parser = std::dynamic_pointer_cast<T>(it->second);
                ERL_DEBUG_ASSERT_PTR(parser);
                return parser;
            }
            auto parser = std::make_shared<T>(option_name, this, member_ptr_in);
            ERL_DEBUG_ASSERT_PTR(parser);
            option_parsers[option_name] = parser;
            return parser;
        }

        void
        Parse() {
            if (args.empty()) { return; }
            auto parsed = po::command_line_parser(args).options(desc).allow_unregistered().run();

            bool any_parsed_options = false;
            for (const auto &option: parsed.options) {
                if (!option.unregistered && option.position_key == -1) {
                    any_parsed_options = true;
                    break;
                }
            }
            if (!any_parsed_options) { return; }  // nothing parsed, skip

            po::store(parsed, vm);
            po::notify(vm);
            vm.clear();  // clear variables_map to allow reparsing and avoid notifying twice.
            args = po::collect_unrecognized<char>(parsed.options, po::include_positional);
        }

        void
        RecordError(const std::string &option_name, const std::string &message) {
            if (!error_msgs.empty()) { return; }  // only record the first error
            error_msgs = fmt::format("Error parsing option {}: {}", option_name, message);
        }

        [[nodiscard]] bool
        Successful() const {
            return error_msgs.empty();
        }
    };

    // considering YAML has broader support for types, we only use boost::program_options to parse
    // the command line into strings. Then we use YAML to convert the strings into the desired
    // types.

    // default specialization for general types
    template<typename T>
    struct ParseOption : ParseOptionBase {

        std::string value_string;

        ParseOption(std::string option_name_in, ProgramOptionsData *po_data_in, T *member_ptr_in)
            : ParseOptionBase(std::move(option_name_in), po_data_in, member_ptr_in) {

            T &member = *static_cast<T *>(member_ptr);

            // do not set default_value here to avoid overwriting parsed values because
            // po_data.vm is cleared after each Parse() call.
            if constexpr (std::is_same_v<T, std::string>) {
                if (member.empty()) {
                    docs = "Type: std::string";
                } else {
                    docs = fmt::format("Type: std::string, default: {}", member);
                }
                po_data->desc.add_options()(
                    option_name.c_str(),
                    po::value<std::string>(&value_string)->value_name(value_name),
                    docs.c_str());
            } else if constexpr (std::is_same_v<T, bool>) {
                docs = fmt::format("Type: bool (true/false, 1/0, yes/no), default: {}", member);
                po_data->desc.add_options()(
                    option_name.c_str(),
                    po::value<std::string>(&value_string)->value_name("FLAG"),
                    docs.c_str());
            } else {
                // single line YAML output
                YAML::Emitter emitter;
                emitter.SetIndent(0);
                emitter.SetMapFormat(YAML::Flow);
                emitter.SetSeqFormat(YAML::Flow);
                emitter << YAML::convert<T>::encode(member);
                docs = fmt::format("Type: {}, default: {}", type_name<T>(), emitter.c_str());
                po_data->desc.add_options()(
                    option_name.c_str(),
                    po::value<std::string>(&value_string)->value_name(value_name),
                    docs.c_str());
            }
        }

        void
        Run() override {
            po_data->Parse();

            if (value_string.empty()) { return; }

            T &member = *static_cast<T *>(member_ptr);
            member = YAML::Load(value_string).template as<T>();
        }
    };

    template<typename T>
    struct ParseOption<std::vector<T>> : ParseOptionBase {
        using Vec = std::vector<T>;

        std::vector<std::string> value_strings;

        ParseOption(std::string option_name_in, ProgramOptionsData *po_data_in, Vec *member_ptr_in)
            : ParseOptionBase(std::move(option_name_in), po_data_in, member_ptr_in) {

            docs = fmt::format(
                "Sequence of {} values",
                std::is_same_v<T, std::string> ? std::string("std::string") : type_name<T>());
            if (Vec &member = *member_ptr_in; !member.empty()) {
                docs += ", default: ";
                // single line YAML output
                YAML::Emitter emitter;
                emitter.SetIndent(0);
                emitter.SetMapFormat(YAML::Flow);
                emitter.SetSeqFormat(YAML::Flow);
                emitter << YAML::convert<Vec>::encode(member);
                docs += emitter.c_str();
            }
            po_data->desc.add_options()(
                option_name.c_str(),
                po::value<std::vector<std::string>>(&value_strings)
                    ->multitoken()
                    ->value_name(value_name),
                docs.c_str());
        }

        void
        Run() override {
            po_data->Parse();

            if (value_strings.empty()) { return; }

            std::vector<T> &member = *static_cast<std::vector<T> *>(member_ptr);
            member.clear();
            member.reserve(value_strings.size());
            std::transform(
                value_strings.begin(),
                value_strings.end(),
                std::back_inserter(member),
                [](const std::string &s) { return YAML::Load(s).as<T>(); });
        }
    };

    template<typename T1, typename T2>
    struct ParseOption<std::pair<T1, T2>> : ParseOptionBase {
        ParseOption<T1> first_parser;
        ParseOption<T2> second_parser;

        using T = std::pair<T1, T2>;

        ParseOption(std::string option_name_in, ProgramOptionsData *po_data_in, T *member_ptr_in)
            : ParseOptionBase(std::move(option_name_in), po_data_in, member_ptr_in),
              first_parser(option_name + ".first", po_data, &member_ptr_in->first),
              second_parser(option_name + ".second", po_data, &member_ptr_in->second) {}

        void
        Run() override {
            first_parser.Run();
            second_parser.Run();
        }
    };

    template<typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
    struct ParseOption<Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>>
        : ParseOptionBase {
        using Mat = Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>;

        std::string arg_str;
        std::vector<Scalar_> values;
        long n_rows = Rows_;
        long n_cols = Cols_;

        ParseOption(std::string option_name_in, ProgramOptionsData *po_data_in, Mat *member_ptr_in)
            : ParseOptionBase(std::move(option_name_in), po_data_in, member_ptr_in) {

            auto options = po_data->desc.add_options();

            Mat &member = *member_ptr_in;

            if (member.size() > 0) {
                values.reserve(member.size());
                for (long i = 0; i < member.size(); ++i) { values.push_back(member.data()[i]); }
            }

            if ((member.IsRowMajor && Rows_ == 1) || (!member.IsRowMajor && Cols_ == 1)) {
                docs = member.IsRowMajor ? "Values of the Eigen vector, separated by ','. The "
                                           "order should be row-major"
                                         : "Values of the Eigen vector, separated by ','. The "
                                           "order should be col-major";
            } else {
                docs = member.IsRowMajor ? "Values of the Eigen matrix, separated by ','. The "
                                           "order should be row-major"
                                         : "Values of the Eigen matrix, separated by ','. The "
                                           "order should be col-major";
            }
            if (!values.empty()) {
                docs += ". Default: ";
                // single line YAML output
                YAML::Emitter emitter;
                emitter.SetIndent(0);
                emitter.SetMapFormat(YAML::Flow);
                emitter.SetSeqFormat(YAML::Flow);
                emitter << YAML::convert<std::vector<Scalar_>>::encode(values);
                docs += emitter.c_str();
            }

            options(
                option_name.c_str(),
                po::value<std::string>(&arg_str)->value_name(value_name),
                docs.c_str());

            if (Rows_ == Eigen::Dynamic) {
                n_rows = member.rows();
                options(
                    (option_name + ".rows").c_str(),
                    po::value<long>(&n_rows)->default_value(n_rows)->value_name("ROWS"),
                    "Number of rows for dynamic-size Eigen matrix");
            }
            if (Cols_ == Eigen::Dynamic) {
                n_cols = member.cols();
                options(
                    (option_name + ".cols").c_str(),
                    po::value<long>(&n_cols)->default_value(n_cols)->value_name("COLS"),
                    "Number of columns for dynamic-size Eigen matrix");
            }
        }

        void
        Run() override {

            Mat &member = *static_cast<Mat *>(member_ptr);

            // parse options
            po_data->Parse();

            const std::vector<std::string> element_strs = SplitString(arg_str, ',');
            if (element_strs.empty()) { return; }

            values.clear();
            values.reserve(element_strs.size());
            for (const std::string &s: element_strs) {
                if (s.empty()) { continue; }
                values.push_back(YAML::Load(s).template as<Scalar_>());
            }

            if (values.empty()) { return; }

            // post-processing
            if (Rows_ > 0 && Cols_ > 0) {
                ERL_ASSERTM(
                    po_data->print_help ||
                        (values.size() == static_cast<std::size_t>(Rows_ * Cols_)),
                    "expecting {} values for option {}, got {}",
                    Rows_ * Cols_,
                    option_name,
                    values.size());
                member = Eigen::Map<Mat>(values.data(), Rows_, Cols_);
                return;
            }
            if (Rows_ > 0) {
                ERL_ASSERTM(
                    po_data->print_help || (values.size() % Rows_ == 0),
                    "expecting multiple of {} values for option {} ({} x -1), got {}",
                    Rows_,
                    option_name,
                    Rows_,
                    values.size());
                const int cols = static_cast<int>(values.size()) / Rows_;
                ERL_ASSERTM(
                    po_data->print_help || n_cols <= 0 || cols == n_cols,
                    "mismatched number of columns: {} vs {}",
                    cols,
                    n_cols);
                member.resize(Rows_, cols);
                member = Eigen::Map<Mat>(values.data(), Rows_, cols);
                return;
            }
            if (Cols_ > 0) {
                ERL_ASSERTM(
                    po_data->print_help || (values.size() % Cols_ == 0),
                    "expecting multiple of {} values for option {:s} (-1 x {}), got {}",
                    Cols_,
                    option_name,
                    Cols_,
                    values.size());
                const int rows = static_cast<int>(values.size()) / Cols_;
                ERL_ASSERTM(
                    po_data->print_help || (n_rows <= 0 || rows == n_rows),
                    "mismatched number of rows: {} vs {}",
                    rows,
                    n_rows);
                member.resize(rows, Cols_);
                member = Eigen::Map<Mat>(values.data(), rows, Cols_);
                return;
            }
            ERL_ASSERTM(
                po_data->print_help || (n_rows > 0 && n_cols > 0),
                "For option {} with fully dynamic size (-1 x -1), both rows and cols must be "
                "specified and > 0",
                option_name);
            const int size = static_cast<int>(values.size());
            ERL_ASSERTM(
                po_data->print_help || (size == n_rows * n_cols),
                "expecting {} values for option {} ({} x {}), got {}",
                n_rows * n_cols,
                option_name,
                n_rows,
                n_cols,
                size);
            member.resize(n_rows, n_cols);
            member = Eigen::Map<Mat>(values.data(), n_rows, n_cols);
        }
    };

    #ifdef ERL_USE_OPENCV

    template<>
    struct ParseOption<cv::Scalar> : ParseOptionBase {
        std::vector<double> values;

        ParseOption(
            std::string option_name_in,
            ProgramOptionsData *po_data_in,
            cv::Scalar *member_ptr_in)
            : ParseOptionBase(std::move(option_name_in), po_data_in, member_ptr_in) {

            cv::Scalar &member = *static_cast<cv::Scalar *>(member_ptr);
            docs = fmt::format(
                "Values of the cv::Scalar (up to 4 values), default: [{}, {}, {}, {}]",
                member[0],
                member[1],
                member[2],
                member[3]);
            po_data->desc.add_options()(
                option_name.c_str(),
                po::value<std::vector<double>>(&values)->multitoken()->value_name(value_name),
                docs.c_str());
        }

        void
        Run() override {

            po_data->Parse();

            if (values.empty()) { return; }

            ERL_ASSERTM(
                po_data->print_help || values.size() <= 4,
                "expecting up to 4 values for option {}, got {}",
                option_name,
                values.size());

            cv::Scalar &member = *static_cast<cv::Scalar *>(member_ptr);
            for (size_t i = 0; i < values.size(); ++i) { member[static_cast<int>(i)] = values[i]; }
        }
    };

    #endif  // ERL_USE_OPENCV

}  // namespace erl::common::program_options

    #define ERL_PARSE_BOOST_OPTION_ENUM(T)                                                \
        template<>                                                                        \
        struct erl::common::program_options::ParseOption<T>                               \
            : erl::common::program_options::ParseOptionBase {                             \
            using yaml_convert = YAML::convert<T>;                                        \
                                                                                          \
            std::string option_str;                                                       \
                                                                                          \
            ParseOption(                                                                  \
                std::string option_name_in,                                               \
                ProgramOptionsData *po_data_in,                                           \
                T *member_ptr_in)                                                         \
                : ParseOptionBase(std::move(option_name_in), po_data_in, member_ptr_in) { \
                docs = "Options:";                                                        \
                for (const auto &enum_member: yaml_convert::EnumSchema) {                 \
                    docs += fmt::format(" {}", enum_member.name);                         \
                }                                                                         \
                option_str = YAML::Dump(yaml_convert::encode(*member_ptr_in));            \
                docs += ", default: ";                                                    \
                docs += option_str;                                                       \
                po_data->desc.add_options()(                                              \
                    option_name.c_str(),                                                  \
                    po::value<std::string>(&option_str)->value_name(value_name),          \
                    docs.c_str());                                                        \
            }                                                                             \
            void                                                                          \
            Run() override {                                                              \
                po_data->Parse();                                                         \
                T &member = *static_cast<T *>(member_ptr);                                \
                yaml_convert::decode(YAML::Node(option_str), member);                     \
            }                                                                             \
        }

#else
    #define ERL_PARSE_BOOST_OPTION_ENUM(T)
#endif  // ERL_USE_BOOST
