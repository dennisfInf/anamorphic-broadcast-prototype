#!/usr/bin/env python3
"""Regenerate explicit-coin vectors from the independent Python reference.

No Python/Rust random-stream equivalence is assumed. These small fixed test keys
are public test data. Run with an explicit reference module path when developing
against a different reference copy.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import struct
import sys


def load_reference(path):
    spec = importlib.util.spec_from_file_location("one_ciphertext_reference", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load reference module: {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclass resolves annotations through its defining module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=here.parents[2] / "research/history/prototypes/python-reference/basic/one_ciphertext.py")
    parser.add_argument("--output", type=Path, default=here / "fixtures" / "python.json")
    args = parser.parse_args()
    ref = load_reference(args.reference.resolve())
    p = ref.Parameters(n=6, m=28, q=257, noise_sd=0.25, n_min=2, L=2)
    ordinary = ref.keygen(p, random.Random(20261012))
    pk = ordinary.public
    expanded = ref.anamorphic_keygen(pk, 3, random.Random(19051))
    seeded = ref.anamorphic_keygen(pk, 3, random.Random(9875), compressed=True)
    sources = {"expanded": expanded, "seeded": seeded}
    fixture = {
        "format": "basic-anamorphic-explicit-coins-v1",
        "params": asdict(p),
        "matrix": [x for row in pk.matrix for x in row],
        "vector": pk.vector,
        "ordinary_key": ordinary.secret,
        "key_sets": {
            name: {
                "user_keys": [list(k) for k in keys.user_keys],
                "blocks": [ref.user_blocks(k, p) for k in keys.user_keys],
                "directory": keys.directory,
            }
            for name, keys in sources.items()
        },
        "cases": [],
        "seed_vectors": [],
    }

    def add_case(name, bit, secret, errors, scalar_error, **extra):
        ct = ref.native_from_coins(pk, bit, secret, errors, scalar_error)
        ordinary_residual = ref.ordinary_residual(ct, ordinary.secret, p.q)
        assert ordinary_residual == (bit * p.delta + scalar_error -
                                     ref.dot(ordinary.secret, errors)) % p.q
        assert ref.ordinary_decrypt(ct, ordinary.secret, p.q) == bit
        fixture["cases"].append({
            "name": name, "bit": bit, "secret": secret, "errors": errors,
            "scalar_error": scalar_error, "fields": ct.fields(),
            "bytes": list(ref.serialize(ct, p)),
            "ordinary_residual": ordinary_residual,
            "hidden_residuals": {
                source: [ref.hidden_residuals(ct, key, p) for key in keys.user_keys]
                for source, keys in sources.items()
            }, **extra,
        })

    errors = tuple(j % 5 - 2 for j in range(p.m))
    for bit in (0, 1):
        add_case(f"ordinary-{bit}", bit, tuple((17 * j + 239) % p.q for j in range(p.n)),
                 errors, -3)

    for source, keys in sources.items():
        for audience in ((2,), (2, 0)):
            plan = ref.prepare_hidden(pk, keys.directory, audience)
            assert plan.fiber.full_rank
            for payload in itertools.product((0, 1), repeat=p.L):
                target = tuple(p.delta * b for _ in plan.audience for b in payload)
                free_values = tuple((61 * j + 251) % p.q for j in range(len(plan.fiber.free)))
                secret = plan.fiber.from_free(target, free_values)
                bit = sum(payload) % 2
                ct = ref.native_from_coins(pk, bit, secret, errors, -3)
                for recipient in plan.audience:
                    assert ref.hidden_decrypt(ct, keys.user_keys[recipient], p) == payload
                add_case(f"{source}-{list(audience)}-{list(payload)}", bit, secret, errors, -3,
                         source=source, audience_input=audience, audience=plan.audience,
                         payload=payload, constraints=plan.fiber.matrix,
                         pivots=plan.fiber.pivots, free_columns=plan.fiber.free,
                         reduced=plan.fiber.reduced, transform=plan.fiber.transform,
                         target=target, free_values=free_values)

    for seed_params, seed in ((p, bytes(range(32))),
                              (ref.Parameters(n=6, m=13, q=257, noise_sd=0.25,
                                              n_min=2, L=3), bytes([91]) * 32)):
        input_bytes = (b"one-ciphertext-anamorphic-broadcast/seed-expansion/v1\0" +
                       struct.pack("<QQQ", seed_params.n, seed_params.m, seed_params.L) + seed)
        raw = hashlib.shake_256(input_bytes).digest((seed_params.L * seed_params.m + 7) // 8)
        fixture["seed_vectors"].append({"params": asdict(seed_params), "seed": list(seed),
                                        "shake_bytes": list(raw),
                                        "blocks": ref.expand_seed(seed, seed_params)})
    assert bytes(fixture["seed_vectors"][0]["shake_bytes"]).hex() == "4740580e5c9bea"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {len(fixture['cases'])} explicit-coin cases to {args.output}")


if __name__ == "__main__":
    main()
