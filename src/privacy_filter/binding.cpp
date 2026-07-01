// nanobind glue over the privacy-filter.cpp flat C API (extern/.../include/pf.h).
//
// Design notes:
//  - The C API returns UTF-8 *byte* offsets; a Python str converts to a
//    std::string (UTF-8) via nanobind, so text.size()/offsets line up.
//  - Every malloc'd buffer from the C side is copied into Python objects and
//    freed here (pf_entities_free / pf_buf_free) before returning.
//  - `label` points into ctx-owned memory valid only until pf_free; we copy it.
//  - Inference calls release the GIL.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include <string>
#include <vector>

#include "pf.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// A detected PII span. Offsets are UTF-8 byte offsets into the source text.
struct Entity {
    int32_t     start;
    int32_t     end;
    float       score;
    std::string label;

    // Convenience: the matched substring, given the original text.
    std::string text(const std::string & source) const {
        if (start < 0 || end > (int32_t) source.size() || start > end) {
            throw nb::value_error("entity offsets out of range for source text");
        }
        return source.substr((size_t) start, (size_t) (end - start));
    }
};

class PrivacyFilter {
public:
    PrivacyFilter(const std::string & gguf_path, const std::string & device, int n_threads) {
        if (pf_abi_version() != PF_ABI_VERSION) {
            throw std::runtime_error("pf ABI version mismatch between header and library");
        }
        ctx_ = pf_load(gguf_path.c_str(), device.empty() ? nullptr : device.c_str(), n_threads);
        if (!ctx_) {
            throw std::runtime_error("pf_load failed (returned NULL) for: " + gguf_path);
        }
        const char * err = pf_last_error(ctx_);
        if (err && *err) {
            std::string msg = err;
            pf_free(ctx_);
            ctx_ = nullptr;
            throw std::runtime_error("pf_load: " + msg);
        }
    }

    ~PrivacyFilter() { close(); }

    // Owns a raw ctx_; copying would double-free it. Keep ownership unique.
    PrivacyFilter(const PrivacyFilter &) = delete;
    PrivacyFilter & operator=(const PrivacyFilter &) = delete;

    void close() {
        if (ctx_) {
            pf_free(ctx_);
            ctx_ = nullptr;
        }
    }

    void set_window(int32_t max_forward_tokens) {
        require_open();
        pf_set_window(ctx_, max_forward_tokens);
    }

    std::vector<Entity> classify(const std::string & text, float threshold) {
        require_open();
        pf_entity * ents = nullptr;
        size_t n = 0;
        int rc;
        {
            nb::gil_scoped_release release;
            rc = pf_classify(ctx_, text.data(), text.size(), threshold, &ents, &n);
        }
        if (rc != 0) {
            throw std::runtime_error(std::string("pf_classify: ") + err_str());
        }
        std::vector<Entity> out;
        out.reserve(n);
        for (size_t i = 0; i < n; i++) {
            out.push_back(Entity{ents[i].start, ents[i].end, ents[i].score,
                                  ents[i].label ? std::string(ents[i].label) : std::string()});
        }
        pf_entities_free(ents, n);
        return out;
    }

    // Returns (ids, offsets) where offsets[i] = (start_byte, end_byte).
    std::tuple<std::vector<int32_t>, std::vector<std::tuple<int32_t, int32_t>>>
    tokenize(const std::string & text) {
        require_open();
        int32_t * ids = nullptr;
        int32_t * offs = nullptr;
        size_t n = 0;
        int rc;
        {
            nb::gil_scoped_release release;
            rc = pf_tokenize(ctx_, text.data(), text.size(), &ids, &offs, &n);
        }
        if (rc != 0) {
            throw std::runtime_error(std::string("pf_tokenize: ") + err_str());
        }
        std::vector<int32_t> id_vec(ids, ids + n);
        std::vector<std::tuple<int32_t, int32_t>> off_vec;
        off_vec.reserve(n);
        for (size_t i = 0; i < n; i++) {
            off_vec.emplace_back(offs[2 * i], offs[2 * i + 1]);
        }
        pf_buf_free(ids);
        pf_buf_free(offs);
        return {std::move(id_vec), std::move(off_vec)};
    }

    const char * last_error() const { return ctx_ ? pf_last_error(ctx_) : nullptr; }

private:
    void require_open() const {
        if (!ctx_) throw std::runtime_error("PrivacyFilter is closed");
    }
    const char * err_str() const {
        const char * e = pf_last_error(ctx_);
        return (e && *e) ? e : "unknown error";
    }

    pf_ctx * ctx_ = nullptr;
};

} // namespace

NB_MODULE(_core, m) {
    m.doc() = "Python bindings for privacy-filter.cpp (PII/NER token classification)";
    m.attr("__abi_version__") = PF_ABI_VERSION;
    m.def("abi_version", &pf_abi_version, "Runtime ABI version of the linked pf library.");

    nb::class_<Entity>(m, "Entity",
                       "A detected PII span. start/end are UTF-8 byte offsets into the source text.")
        .def_ro("start", &Entity::start)
        .def_ro("end", &Entity::end)
        .def_ro("score", &Entity::score)
        .def_ro("label", &Entity::label)
        .def("text", &Entity::text, "source"_a,
             "Return the matched substring given the original text.")
        .def("__repr__", [](const Entity & e) {
            return "Entity(start=" + std::to_string(e.start) +
                   ", end=" + std::to_string(e.end) +
                   ", score=" + std::to_string(e.score) +
                   ", label='" + e.label + "')";
        });

    nb::class_<PrivacyFilter>(m, "PrivacyFilter",
                              "Loaded privacy-filter model. Use as a context manager or call close().")
        .def(nb::init<const std::string &, const std::string &, int>(),
             "gguf_path"_a, "device"_a = "cpu", "n_threads"_a = 0,
             "Load a GGUF model. device: 'cpu'|'gpu'|'cuda'|'vulkan' (optionally ':N'). "
             "n_threads <= 0 picks a default (CPU only).")
        .def("classify", &PrivacyFilter::classify, "text"_a, "threshold"_a = 0.0f,
             "Detect PII entities in text. Entities scoring below threshold are dropped.")
        .def("tokenize", &PrivacyFilter::tokenize, "text"_a,
             "Tokenize text; returns (ids, offsets) with byte-offset (start, end) pairs.")
        .def("set_window", &PrivacyFilter::set_window, "max_forward_tokens"_a,
             "Set max tokens per forward pass (default 4096); must be > 2048 to window.")
        .def("close", &PrivacyFilter::close, "Free the model. Idempotent.")
        // Return the same Python object (not a copy that would share ctx_).
        .def("__enter__", [](nb::object self) { return self; })
        .def("__exit__", [](PrivacyFilter & self, nb::args) { self.close(); });
}
