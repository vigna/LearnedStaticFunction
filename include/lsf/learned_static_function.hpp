#pragma once

#include "filtered_lsf_storage.hpp"
#include "dataset_reader.hpp"
#include "model_wrapper.hpp"

namespace lsf {

    template<typename DataSet, typename Model, typename Storage>
    class LearnedStaticFunction {
        XXH3_state_t *state;
        Model &model;
        Storage storage;

    public:

        LearnedStaticFunction(const DataSet &dataset, Model &model) : model(model) {
            state = XXH3_createState();
            assert(state);
            storage = Storage();
            storage.build(
                    dataset.size(),
                    dataset.classes_count(),
                    [&](size_t i) {
                        auto example = dataset.get_example(i);
                        return std::make_tuple(hash(i, example), dataset.get_label(i), model.invoke(example));
                    });

            std::cout << "Model size: " << model_bytes() * 8 << " bits\n";
            std::cout << "Total size: " << size_in_bytes() * 8 << " bits\n";
            std::cout << "Total bits/example: " << (size_in_bytes() * 8 / static_cast<double>(dataset.size())) << "\n";
        }

        std::span<float> query_probabilities(std::span<const float> features) {
            return model.invoke(features);
        }

        auto query_storage(uint64_t key, std::span<const float> features) {
            return storage.query_storage(hash(key, features));
        }

        uint64_t query(uint64_t key, std::span<const float> features) {
            return storage.query(hash(key, features), query_probabilities(features));
        }

        size_t model_bytes() const { return model.model_bytes(); }

        size_t storage_bytes() const { return storage.size_in_bytes(); }

        size_t size_in_bytes() const { return storage.size_in_bytes() + model.model_bytes(); }

        size_t get_statistic_bits_input() const { return storage.get_statistic_bits_input(); }

    private:

        uint64_t hash(uint64_t key, std::span<const float> features) {
            XXH3_64bits_reset(state);
            XXH3_64bits_update(state, &key, sizeof(size_t));
            //XXH3_64bits_update(state, features.data(), features.size_bytes());
            return XXH3_64bits_digest(state);
        }
    };
}