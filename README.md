# Learned Static Function Data Structures, Revisited

This repository is a fork of the [original LearnedStaticFunction
repository](https://github.com/gvinciguerra/LearnedStaticFunction) for the paper
“[Learned Static Function Data
Structures](https://doi.org/10.14778/3796195.3796205)” by Stefan Hermann,
Hans-Peter Lehmann, Giorgio Vinciguerra, and Stefan Walzer (PVLDB 19(5), 2026).
The paper's main baseline is GOV, the compressed static function from my paper
with Marco Genuzio and Giuseppe Ottaviano “[Fast scalable construction of
([compressed] static | minimal perfect hash)
functions](https://doi.org/10.1016/j.ic.2020.104517)”, implemented in
[Sux4J](https://github.com/vigna/Sux4J). This fork contains a fix to the GOV
benchmark, which has since been merged upstream. The original README is in
[`README-orig.md`](README-orig.md).

If you're looking for advice on the choice of a static function, the comparison
with GOV in the published paper is misleading: GOV was benchmarked under
conditions different from those of every other structure in the paper. In the
published Table 5, GOV queries are 1.4× to 3.6× slower than those of the
authors' non-learned compressed static function, “Ours (CSF)”. When all
structures are queried in the same way, GOV queries are as fast or faster on
every dataset, by up to 2.3×.

## What happened

Each dataset in the paper is a list of examples, and each example has a feature
vector and a value. In the benchmarks, the key of the *i*-th example is its index
*i*. A learned static function needs the features to evaluate its model; a
static function needs only the key.

The query methods of the paper's data structure,
[`LearnedStaticFunction`](https://github.com/gvinciguerra/LearnedStaticFunction/blob/1781876d3aaaf532e5a732cd841ab966734d0c67/include/lsf/learned_static_function.hpp#L127-L180),
take both a key and a feature vector, and pass both to a private hash method,
which is also used at construction time. Everywhere the class computes a hash,
it passes the features along
([construction](https://github.com/gvinciguerra/LearnedStaticFunction/blob/1781876d3aaaf532e5a732cd841ab966734d0c67/include/lsf/learned_static_function.hpp#L135-L150),
[queries](https://github.com/gvinciguerra/LearnedStaticFunction/blob/1781876d3aaaf532e5a732cd841ab966734d0c67/include/lsf/learned_static_function.hpp#L156-L162)):

```cpp
LearnedStaticFunction(const DataSet &dataset, Model &model) : model(model) {
    // ...
    storage.build(
            dataset.size(),
            dataset.classes_count(),
            [&](size_t i) {
                auto example = dataset.get_example(i);
                return std::make_tuple(hash(i, example), dataset.get_label(i), model.invoke(example));
            });
    // ...
}

// ...

auto query_storage(uint64_t key, std::span<const float> features) {
    return storage.query_storage(hash(key, features));
}

uint64_t query(uint64_t key, std::span<const float> features) {
    return storage.query(hash(key, features), query_probabilities(features));
}
```

The line that feeds the features to the hash function, however, is [commented
out](https://github.com/gvinciguerra/LearnedStaticFunction/blob/1781876d3aaaf532e5a732cd841ab966734d0c67/include/lsf/learned_static_function.hpp#L174-L179)
in the body of the private method:

```cpp
uint64_t hash(uint64_t key, std::span<const float> features) {
    XXH3_64bits_reset(state);
    XXH3_64bits_update(state, &key, sizeof(size_t));
    //XXH3_64bits_update(state, features.data(), features.size_bytes());
    return XXH3_64bits_digest(state);
}
```

All structures built on this class therefore hash an integer and nothing else.
This includes “Ours (CSF)”, whose
[model](https://github.com/gvinciguerra/LearnedStaticFunction/blob/1781876d3aaaf532e5a732cd841ab966734d0c67/include/lsf/model_freq.hpp#L27-L29)
returns the global value frequencies and ignores its argument, so its
[queries](https://github.com/gvinciguerra/LearnedStaticFunction/blob/1781876d3aaaf532e5a732cd841ab966734d0c67/ribbon_learned_bench.cpp#L103-L109)
never read the features at all. BuRR, the other non-learned competitor, is
[queried directly with the integer
index](https://github.com/gvinciguerra/LearnedStaticFunction/blob/1781876d3aaaf532e5a732cd841ab966734d0c67/ribbon_learned_bench.cpp#L177).

GOV was treated differently. The [training
script](https://github.com/gvinciguerra/LearnedStaticFunction/blob/1781876d3aaaf532e5a732cd841ab966734d0c67/train/train.py#L137-L144)
wrote GOV's keys as the raw bytes of the feature vector (four bytes per feature)
followed by a base85 encoding of the index: about 220 bytes per key on
covertype, 110 on songs, 65 on urls, 45 on nids, and 9 on gauss. The index alone
would have been enough to make the keys distinct. The [query
benchmark](https://github.com/gvinciguerra/LearnedStaticFunction/blob/1781876d3aaaf532e5a732cd841ab966734d0c67/csf/benchmark.cpp#L112-L118)
loaded the keys into a `std::vector<std::string>`, so each query fetched a
heap-allocated string from a random memory location and hashed all its bytes.
GOV was the only structure in the comparison that hashed the features, and the
only one that had to fetch its key from memory before hashing it. Its query
times in Table 5 measure mostly string hashing and memory latency.

The repository history shows how this setup came about. The prototypes added in
May 2025
([5731cdf](https://github.com/gvinciguerra/LearnedStaticFunction/commit/5731cdffaf524027a0d9c5a18d8d1b2fd5a3383a))
hashed the index together with the features. The commit that introduced the
current interface on July 2, 2025
([5c5d895](https://github.com/gvinciguerra/LearnedStaticFunction/commit/5c5d8953d5ce2163b41fb23a251e96d8011fe47d))
already has the feature line commented out, and the prototypes were deleted the
next day
([55efa4e](https://github.com/gvinciguerra/LearnedStaticFunction/commit/55efa4ebcf594def39f07181a17e73b0536fefda)).
Three weeks later, on July 22, 2025, the GOV benchmark was added
([d2d1d34](https://github.com/gvinciguerra/LearnedStaticFunction/commit/d2d1d3411f8d2090b52e08efe1b2081d09250e43),
[5d57bfa](https://github.com/gvinciguerra/LearnedStaticFunction/commit/5d57bfaa03b1031c26fa203e750ef4c69ad23bd0)),
with keys containing the features.

## How it was fixed

In April 2026 I [opened an
issue](https://github.com/gvinciguerra/LearnedStaticFunction/issues/1) asking
why GOV and BuRR were queried with different keys: a GOV built on the songs
dataset with the same integer keys and values given to BuRR answers queries in
27 ns on my laptop, against the 269 ns reported in the paper. I then submitted a
[pull request](https://github.com/gvinciguerra/LearnedStaticFunction/pull/2)
giving GOV the same keys as every other structure (the index, as a 4-byte
integer), and the authors merged it on April 30, 2026. By then, however, the
[PVLDB version](https://www.vldb.org/pvldb/vol19/p917-vinciguerra.pdf),
finalized in February, already contained the original data.

The table below compares the query times (µs/key) of GOV and “Ours (CSF)” in the
published paper and in the [corrected arXiv
version](https://arxiv.org/abs/2510.27588v3). The corrected numbers come from a
full rerun, so all timings differ slightly; only those of GOV change
substantially, by 2.4× to 5.4×.

| Dataset   | Ours (CSF), published | GOV, published | Ours (CSF), corrected | GOV, corrected |
| --------- | --------------------: | -------------: | --------------------: | -------------: |
| covertype |                 0.125 |          0.342 |                 0.115 |          0.063 |
| nids      |                 0.149 |          0.244 |                 0.133 |          0.059 |
| songs     |                 0.156 |          0.269 |                 0.134 |          0.062 |
| urls      |                 0.052 |          0.186 |                 0.043 |          0.042 |
| gauss     |                 0.141 |          0.202 |                 0.105 |          0.084 |

The space figures are unaffected, and the authors' CSF remains smaller than GOV.
It is also faster to build, but this comparison is not a fair one either. As the
paper notes, “the GOV construction is implemented in Java, but the queries are
executed in C++”: GOV is built by a generic Java implementation that turns keys
into bit vectors through a transformation strategy chosen at run time, whereas
the authors' structures are built by C++ code specialized at compile time.

Three claims of the published paper do not survive the correction:

- “Our CSF outperforms the state-of-the-art GOV competitor in all three metrics
  simultaneously on all datasets.” GOV answers queries faster on every dataset
  except urls, where the two are tied.

- “On all datasets, query times are at most 17.0× slower than GOV.” With the
  corrected numbers, the learned static function on songs (L2 U50) is 56× slower
  than GOV (3.478 vs. 0.062 µs).

- “The query time of our LSF on the gauss dataset σ = 0.25 is close to the GOV
  competitor.” It is 3.6× slower (0.300 vs. 0.084 µs).

## Conclusions

The authors have not explained how the asymmetry arose.

The authors' structures were given an interface that takes the features, and
their earlier prototypes hashed them; the feature hashing was disabled when the
interface was written. The GOV benchmark, written three weeks later, was fed
keys containing the features, although the index alone would have sufficed. GOV
is the only structure in Table 5 that none of the paper's authors co-authored,
and it is the only one that hashed the features. The resulting asymmetry
supported one of the paper's claims: that the authors' non-learned CSF beats the
state of the art in all three metrics.

The published GOV timings were implausible: GOV queries were 3.8× to 17× slower
than those of BuRR, whereas with the same keys the gap is between 1.9× and 4.2×.
The commits are public, and readers can judge for themselves.

## Corrected versions

A corrected version of the paper is available on arXiv as
[v3](https://arxiv.org/abs/2510.27588v3) (May 19, 2026). It contains the
corrected Table 5 and revises the sentences quoted above; the statement about
the maximum query slowdown with respect to GOV is replaced by an average over
the fastest learned variants (12.2×). After I contacted the editors-in-chief of
PVLDB, explaining the situation and providing corrected numbers, they asked the
authors for an erratum, which will soon be published in the journal. The
authors had not offered one on their own.

## Reproducing the results

The Docker image described in [`README-orig.md`](README-orig.md) runs the whole
pipeline: training, all benchmarks (including the corrected GOV benchmark), and
the generation of the paper's tables. It takes several hours.
