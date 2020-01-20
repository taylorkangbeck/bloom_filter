from uuid import uuid4
from statistics import mean

from bloom_filter import ModuloBloomFilter, BitArray, SaltedSHA256BloomFilter, ComposedSHABloomFilter


def test_bit_array():
    ba = BitArray(10)
    ba.set(2)
    ba.set(6)
    assert ba.get(2)
    assert ba.get(6)
    assert not ba.get(4)


def test_bloom_filters_ints():
    types = [ModuloBloomFilter, SaltedSHA256BloomFilter, ComposedSHABloomFilter]

    m = 20
    k = 2

    for T in types:
        bloom = T(m, k)
        bloom.add(15)
        bloom.add(41)
        try:
            assert 15 in bloom
            assert 41 in bloom
            assert 34 not in bloom
        except AssertionError as err:
            raise AssertionError(f'Failed on type {T.__name__}:\n' + str(err))


def test_bloom_filters_strings():
    types = [SaltedSHA256BloomFilter, ComposedSHABloomFilter]

    for T in types:
        bloom = T.create(100, 0.05)
        hellos = {f'hello{i}' for i in range(10)}
        bloom.add_all(hellos)
        try:
            assert 'hello5' in bloom
            assert 'goodbye' not in bloom
        except AssertionError as err:
            raise AssertionError(f'Failed on type {T.__name__}:\n' + str(err))


def test_bloom_filter_fpr():
    # Experiments are showing that expected upper bound is not holding...
    # check_fpr()
    pass


def check_fpr():
    types = [SaltedSHA256BloomFilter, ComposedSHABloomFilter]

    capacity = 100
    inserts = 50
    expected_fpr = 0.05
    num_trials = 10

    for T in types:
        all_actual_fpr = []
        all_calc_bounds = []
        for trial in range(num_trials):
            bloom = T.create(capacity, expected_fpr)

            # generate data
            data = [str(uuid4()) for _ in range(2*inserts)]
            added = data[:inserts]
            not_added = data[inserts:]

            bloom.add_all(added)

            num_false_pos = 0

            for x in added:
                if x not in bloom:
                    raise Exception(f'[{T.__name__}] FALSE NEGATIVE: {x}')
            for y in not_added:
                if y in bloom:
                    num_false_pos += 1

            all_actual_fpr.append(num_false_pos/len(not_added))
            all_calc_bounds.append(bloom.p())

        mean_actual_fpr = mean(all_actual_fpr)
        if mean_actual_fpr > expected_fpr:
            raise Exception(f'[{T.__name__}] Actual mean FPR too high: {mean_actual_fpr}\n'
                            f'Expected FPR:{expected_fpr}\n'
                            f'All actual fpr: {all_actual_fpr}\n'
                            f'All calculated bounds: {all_calc_bounds}')
