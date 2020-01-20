import abc
from math import ceil, log, exp
import hashlib


class BitArray:
    def __init__(self, size):
        self._bits = 1 << size
        self.size = size

    def _val(self, i):
        if not 0 <= i < self.size:
            raise ValueError(f'Index {i} must be between 0 and bit array size {self.size}')

    def get(self, i):
        self._val(i)
        return self._bits & 1 << i > 0

    def set(self, i):
        self._val(i)
        self._bits = self._bits | 1 << i

    def unset(self, i):
        self._val(i)
        self._bits = self._bits ^ 1 << i


class BloomFilterBase(abc.ABC):
    def _create_bit_array(self, size):
        self._bit_array = BitArray(size)

    @abc.abstractmethod
    def _get_hash_funcs(self, k):
        ...

    def __init__(self, m, k):
        """
        Constructs a bloom filter using an underlying bit array of size m and k hash functions.
        Use create() instead to ensure a bounded false positive rate.
        :param m: size of bit array
        :param k: number of hash functions to use
        """
        self._create_bit_array(m)
        self._hash_funcs = self._get_hash_funcs(k)
        assert len(self._hash_funcs) == k
        self._num_added = 0

    def __contains__(self, item):
        for hash_func in self._hash_funcs:
            if not self._bit_array.get(hash_func(item) % self.m()):
                return False
        return True

    @classmethod
    def create(cls, capacity, false_positive_rate):
        """
        Returns a bloom filter that ensures a maximum false positive rate up to a certain capacity.
        :param capacity: number of elements that can be added before exceeding false positive rate bound
        :param false_positive_rate: the maximum false positive rate for capacity elements
        :return:
        """
        if not 0 < false_positive_rate < 1:
            raise ValueError('FPR must be between 0 and 1')

        n = capacity
        p = false_positive_rate

        # From: https://en.wikipedia.org/wiki/Bloom_filter#Optimal_number_of_hash_functions
        m = ceil(-(n * log(p)) / (log(2) ** 2))
        k = ceil((m / n) * log(2))

        while cls._calc_fpr_bound(k, n, m) > p:
            m += 1
            k = ceil((m / n) * log(2))

        bloom = cls(m, k)
        bloom.capacity = capacity
        return bloom

    def add(self, item):
        for hash_func in self._hash_funcs:
            self._bit_array.set(hash_func(item) % self.m())
        self._num_added += 1

    def add_all(self, items):
        for item in items:
            self.add(item)

    @staticmethod
    def _calc_fpr_bound(k, n, m):
        # Using Goel and Gupta upper bound
        # From: https://en.wikipedia.org/wiki/Bloom_filter#Optimal_number_of_hash_functions
        return (1 - (exp(-(k * (n + 0.5)) / (m - 1)))) ** k

    @staticmethod
    def _approx_fpr(k, n, m):
        return (1 - (exp(-(k * n) / m))) ** k

    # Exposing commonly used Bloom filter values
    def k(self):
        # number of hash functions
        return len(self._hash_funcs)

    def m(self):
        # size of bit array
        return self._bit_array.size

    def n(self):
        # number of items added to bloom filter
        return self._num_added

    def p(self):
        # false positive probability
        return self._calc_fpr_bound(self.k(), self.n(), self.m())


class ModuloBloomFilter(BloomFilterBase):
    # for testing
    def _get_hash_funcs(self, k):
        return [lambda x: x % self._bit_array.size for _ in range(k)]


class SaltedSHA256BloomFilter(BloomFilterBase):
    # Composing hash functions by adding a salt to SHA256 input
    @staticmethod
    def _sha256(x):
        return int(hashlib.sha256(str(x).encode()).hexdigest(), 16)

    def _get_hash_funcs(self, k):
        return [lambda x: self._sha256(f'{str(x)}-{i}') for i in range(k)]


class ComposedSHABloomFilter(BloomFilterBase):
    # Composing hash functions from SHA256 and SHA224
    # Using method from https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf
    @staticmethod
    def _sha256(x):
        return int(hashlib.sha256(str(x).encode()).hexdigest(), 16)

    @staticmethod
    def _sha224(x):
        return int(hashlib.sha224(str(x).encode()).hexdigest(), 16)

    def _get_hash_funcs(self, k):
        return [lambda x: self._sha224(x) + i * self._sha256(x) for i in range(k)]
