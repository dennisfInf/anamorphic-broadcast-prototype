#!/usr/bin/env python3
"""Regenerate public, deterministic cross-language equation and sampler fixtures.

Run with the project Python environment (numpy and python-flint installed):
  python -B tests/regenerate_fixtures.py --reference /path/to/practical-run
The reference imports never load runtime.py or compile its C kernels. Explicit
coins test algebra, not a shared production RNG stream or a parameter claim.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace


def matrix_values(matrix):
    return [[int(matrix[i, j]) for j in range(matrix.ncols())]
            for i in range(matrix.nrows())]


def literal_sign_product(y, raw, columns):
    """Independent interpretation of raw[group * columns + output]."""
    if len(raw) != ((len(y) + 7) // 8) * columns:
        raise ValueError('wrong coin length')
    return [sum(y[i] * (1 if raw[(i // 8) * columns + j] >> (i % 8) & 1 else -1)
                for i in range(len(y))) for j in range(columns)]


class WordStream:
    """One MSB-first stream; calls of 64,128,... bits retain the same prefix."""
    def __init__(self, words):
        self.words = words
        self.offset = 0
        self.calls = []

    def getrandbits(self, count):
        if count % 64:
            raise ValueError('reference expected word-aligned refinement')
        value = 0
        for _ in range(count // 64):
            if self.offset == len(self.words):
                raise RuntimeError('fixture random stream exhausted')
            value = value << 64 | self.words[self.offset]
            self.offset += 1
        self.calls.append(count)
        return value


def words_after(prefix, tag):
    rng = random.Random(202610190000 + tag)
    return [prefix] + [rng.getrandbits(64) for _ in range(31)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent / 'fixtures' / 'python.json')
    args = parser.parse_args()
    reference = args.reference.resolve()
    sys.path.insert(0, str(reference / 'reference'))
    sys.path.insert(0, str(reference))
    from flint import fmpq, fmpz_mat
    import proof_implementation as proof
    from proof_sampling import rounded_normal

    rng = random.Random(20261019)
    n, m, J, q, group_size, message_bits = 7, 11, 2, 65537, 2, 2
    g = group_size * message_bits
    params = SimpleNamespace(n=n, m=m, J=J, q=q, g=g)
    A0 = fmpz_mat(n, m, [rng.randrange(q) for _ in range(n*m)])
    A1 = fmpz_mat(n, m, [rng.randrange(q) for _ in range(n*m)])
    D1 = fmpz_mat(m, J, [rng.randrange(-3, 4) for _ in range(m*J)])
    D2 = fmpz_mat(m, J, [1, 0, 0, 1] + [rng.randrange(-3, 4) for _ in range((m-2)*J)])
    kernel = proof.AffineSampler(D2.transpose(), q)
    B = kernel.uniform_kernel_rows(n, rng)
    U = proof.mod_matrix(A0*D1 + A1*D2, q)
    setup = proof.Setup(params, A0, A1, B, U, D1, D2, 1)
    assert all(int(x) % q == 0 for x in (B*D2).entries())
    keys = fmpz_mat(m, g, [rng.getrandbits(1) for _ in range(m*g)])
    public = A0*keys
    constraints = fmpz_mat(g, n, [(int(A0[j, 0])-int(public[j, i])) % q
                                  for i in range(g) for j in range(n)])
    sampler = proof.AffineSampler(constraints, q)
    assert sampler.full_rank, 'fixture seed must produce full-rank example'
    covert = proof.CovertKeys(keys, constraints, sampler, group_size, message_bits)
    raw = [rng.randrange(256) for _ in range(((m+7)//8)*m)]
    y = [(i % 7) - 3 for i in range(m)]
    correlated = literal_sign_product(y, raw, m)
    x = [-2, 1]

    # Literal column sums are intentionally independent of the Rust/C kernels.
    def encryption_case(name, ordinary, secret, yy, xx, zz, target=None, free=None):
        v = [(sum(int(A0[i, j])*secret[i] for i in range(n))+yy[j]) % q for j in range(m)]
        w = [(sum((int(A1[i, j])+int(B[i, j]))*secret[i] for i in range(n))+zz[j]) % q
             for j in range(m)]
        c = [(sum(int(U[i, j])*secret[i] for i in range(n))+xx[j]+(q//2)*ordinary[j]) % q
             for j in range(J)]
        ct = proof.Ciphertext(tuple(v), tuple(w), tuple(c))
        residual = list(proof.ordinary_residual(setup, ct))
        hidden = list(proof.covert_residual(covert, ct, q))
        expected_residual = [(xx[j]+(q//2)*ordinary[j]-sum(int(D1[i,j])*yy[i]+int(D2[i,j])*zz[i]
                                                                    for i in range(m))) % q for j in range(J)]
        assert residual == expected_residual
        assert hidden == [(sum(int(constraints[a,i])*secret[i] for i in range(n))
                           +yy[0]-sum(int(keys[i,a])*yy[i] for i in range(m))) % q for a in range(g)]
        wire = proof.encode_ciphertext(params, ct)
        assert proof.decode_ciphertext(params, wire) == ct
        # The reference writes v || w || c. The crate writes Epoque's CT = (c0, c1) as
        # c0 || c1, where c0 = c and c1 = (v; w) = F_id^T s + (y; R^T y).
        field = (q.bit_length() + 7) // 8
        wire = wire[2 * m * field:] + wire[:2 * m * field]
        return dict(name=name, ordinary_bits=ordinary, target=target, free_values=free, secret=secret,
                    errors=dict(y=[str(z) for z in yy], x=[str(z) for z in xx], correlated=[str(z) for z in zz]),
                    ciphertext=dict(c0=c, c1=v + w), codec_hex=wire.hex(),
                    ordinary_residual=residual, covert_residual=hidden,
                    ordinary_decoded=list(proof.decrypt_ordinary(setup, ct)),
                    covert_decoded=[list(bits) for bits in proof.decrypt_covert(covert, ct, q)])

    cases = [encryption_case('ordinary', [1,0], [rng.randrange(q) for _ in range(n)], y,x,correlated)]
    for index, payload in enumerate(itertools.product((0,1), repeat=message_bits)):
        target = [(q//2)*v for v in payload*group_size]
        free = [rng.randrange(q) for _ in sampler.free]
        secret = list(sampler.from_free(target, free))
        case = encryption_case('anamorphic-' + ''.join(map(str,payload)), [index%2, (index+1)%2],
                               secret, y,x,correlated,target,free)
        assert case['covert_decoded'] == [list(payload)]*group_size
        assert case['ordinary_decoded'] == case['ordinary_bits']
        cases.append(case)
    huge_y = [(-1 if i%2 else 1)*((1 << (80+i))+31*i+1) for i in range(m)]
    huge_x = [-(1<<112)+7, (1<<119)+11]
    huge_correlated = literal_sign_product(huge_y, raw, m)
    cases.append(encryption_case('signed-errors-beyond-i64', [1,1], [rng.randrange(q) for _ in range(n)],
                                 huge_y, huge_x, huge_correlated))

    late_constraints = fmpz_mat([[0,0,2,3,5,7], [0,0,0,0,4,9]])
    late = proof.AffineSampler(late_constraints, 257)
    late_target, late_free = [128,0], [1,2,3,4]
    # m=11 includes a partial final group of three useful sign bits.
    sign_fixtures = []
    for name, yy, columns in [('partial-group', y, m), ('huge-signed', huge_y, m),
                               ('one-column', y, 1)]:
        rr = raw if columns == m else [0xA5, 0xFC]
        sign_fixtures.append(dict(name=name,y=[str(z) for z in yy], columns=columns,
                                  raw=rr, expected=[str(z) for z in literal_sign_product(yy,rr,columns)]))

    normal_specs = [
        (1,1,0), (1,1,(1<<64)-1), (1,1,(1<<63)-1), (1,1,1<<63), (1,1,(1<<63)+1),
        (4,1,0x123456789ABCDEF0), (17,3,0xF123456789ABCDEF),
        (17,3,0x3333333333333333), (17,3,0xCCCCCCCCCCCCCCCC),
        (1<<160,7,0x123456789ABCDEF0), (1<<160,7,0), (1<<160,7,(1<<64)-1),
        (34972416,1,0x6666666666666666), (34972416,1,0x9999999999999999),
        (1,1000000,0x123456789ABCDEF0), (1,1000000,0xFFFFFFFFFFFFFFFE),
        (65537**2,(65*1024)**2,0x123456789ABCDEF0),
        (178282599284737**2,(65*80875)**2,0xABCDEF0123456789),
        (((1<<64)-1)**2,1,0), (((1<<64)-1)**2,1,(1<<64)-1),
        (((1<<64)-1)**2,1,0x123456789ABCDEF0)]
    normal_cases = []
    for index,(numerator,denominator,prefix) in enumerate(normal_specs):
        words = words_after(prefix,100+index)
        stream = WordStream(words)
        expected = rounded_normal(fmpq(numerator,denominator),stream)
        normal_cases.append(dict(variance_numerator=str(numerator),variance_denominator=str(denominator),
                                 words=[str(v) for v in words], expected=str(expected),
                                 consumed_words=stream.offset, calls_bits=stream.calls))

    output = dict(schema=1,provenance='Public deterministic explicit coins; Python arbitrary integers and certified Arb intervals.',
        parameters=dict(n=n,m=m,J=J,q=q,g=g,C=1024,sigma_key=2,
                        error_variance_numerator=str(q*q),error_variance_denominator=str((65*1024)**2),
                        group_size=group_size,message_bits=message_bits,identity=1),
        matrices={name:matrix_values(value) for name,value in dict(A0=A0,A1=A1,B=B,U=U,D1=D1,D2=D2,covert_keys=keys).items()},
        covert=dict(constraints=matrix_values(constraints),pivots=list(sampler.pivots),free=list(sampler.free),
                    transform=matrix_values(sampler.transform),reduced_free=matrix_values(sampler.reduced_free)),
        cases=cases,sign_products=sign_fixtures,
        late_pivot=dict(q=257,constraints=matrix_values(late_constraints),pivots=list(late.pivots),
                        free=list(late.free),transform=matrix_values(late.transform),
                        reduced_free=matrix_values(late.reduced_free),target=late_target,free_values=late_free,
                        secret=list(late.from_free(late_target,late_free))),
        rounded_normal=normal_cases)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(f'Wrote {args.output}: {len(cases)} encryption cases, '
          f'{len(normal_cases)} rounded-normal cases.')

if __name__ == '__main__':
    main()
