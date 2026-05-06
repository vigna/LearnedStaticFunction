#pragma once

#include "filter_coding.hpp"

namespace lsf {

    constexpr size_t recDepth = 2;
    constexpr float slotsPerItem = 0.96;
    struct BuRRConfig
            : public ribbon::RConfig<128, 1, ribbon::ThreshMode::twobit, false, true, false, 0, uint64_t> {
        static constexpr bool kUseVLR = true;
        static constexpr Index kBucketSize = 128;
    };


    template<typename Coding>
    class FilteredLSFStorage {
    public:
        ribbon::ribbon_filter<recDepth, BuRRConfig> correctionVLSF;
        ribbon::ribbon_filter<recDepth, BuRRConfig> filterVLSF;
        Coding coder;

    private:
        size_t statistic_bits_input;
    public:

        FilteredLSFStorage() {}

        template<typename F>
        void build(size_t n, size_t classes_count, F get) {
            statistic_bits_input = 0;
            size_t huffman_bits = 0;
            size_t filter_bits = 0;
            using namespace ribbon;
            IMPORT_RIBBON_CONFIG(BuRRConfig);

            auto [hashCSF, labelCSF, probabilitiesCSF] = get(0);
            coder = Coding(classes_count, probabilitiesCSF);
            size_t maxlenfilter = 0;
            auto inputFilter = std::make_unique<std::pair<Key, ResultRowVLR>[]>(n);
            for (size_t i = 0; i < n; ++i) {
                auto [hash, label, probabilities] = get(i);
                auto [code, filterLength, bitsSet] = coder.encode_once_filter(probabilities, label);
                statistic_bits_input += bitsSet;
                inputFilter[i].first = hash;
                if (filterLength > maxlenfilter)
                    maxlenfilter = filterLength;
                inputFilter[i].second = static_cast<uint64_t>(code) | (uint64_t(1) << filterLength);
                filter_bits += filterLength;
            }

            filterVLSF = ribbon_filter<recDepth, BuRRConfig>(slotsPerItem, 42, maxlenfilter);
            filterVLSF.AddRange(inputFilter.get(), inputFilter.get() + n, true);
            filterVLSF.BackSubst();

            size_t maxlen = 0;
            auto input = std::make_unique<std::pair<Key, ResultRowVLR>[]>(n);
            for (size_t i = 0; i < n; ++i) {
                auto [hash, label, probabilities] = get(i);
                uint64_t filterVal = filterVLSF.QueryRetrieval(hash);
                auto [code, length] = coder.encode_once_corrected_code(probabilities, label, filterVal);
                statistic_bits_input += length;
                input[i].first = hash;
                if (length > maxlen)
                    maxlen = length;
                input[i].second = static_cast<uint64_t>(code) | (uint64_t(1) << length);
                huffman_bits += length;
            }

            correctionVLSF = ribbon_filter<recDepth, BuRRConfig>(slotsPerItem, 42, maxlen);
            correctionVLSF.AddRange(input.get(), input.get() + n);
            correctionVLSF.BackSubst();
        }

        std::pair<uint64_t, uint64_t> query_storage(uint64_t hash) {
            uint64_t corrected_code = correctionVLSF.QueryRetrieval(hash);
            uint64_t filterCode = filterVLSF.QueryRetrieval(hash);
            return {corrected_code, filterCode};
        }

        uint64_t query(uint64_t hash, std::span<float> probabilities) {
            auto [corrected_code, filterCode] = query_storage(hash);
            return coder.decode_once(probabilities, corrected_code, filterCode);
        }

        size_t size_in_bytes() const {
            return filterVLSF.Size() + correctionVLSF.Size();
        }

        size_t get_statistic_bits_input() const {
            return statistic_bits_input;
        }

        static const std::string get_name() {
            return "Filtered-" + Coding::get_name();
        }
    };
}
