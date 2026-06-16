#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from __future__ import annotations

import hashlib
from collections.abc import Callable, Collection, Iterator, MutableMapping, Sequence
from contextlib import contextmanager
from typing import NamedTuple, TypeVar

_GREEDY_TEMPERATURE_EPS = 1e-5
_HASH_DIGEST_SIZE = 8
_MAX_TORCH_GENERATOR_SEED = (1 << 63) - 1

T = TypeVar("T")


class DyCPSampleSeed(NamedTuple):
    req_idx: int
    seed: int


def make_dycp_sample_seed(
    model_seed: int | None,
    req_id: str,
    sample_pos: int,
    phase: str,
) -> int:
    hasher = hashlib.blake2b(digest_size=_HASH_DIGEST_SIZE)
    for value in (int(model_seed or 0), req_id, int(sample_pos), phase):
        hasher.update(str(value).encode("utf-8", errors="surrogatepass"))
        hasher.update(b"\0")
    return int.from_bytes(hasher.digest(), "little") & _MAX_TORCH_GENERATOR_SEED


def iter_dycp_sample_seeds(
    *,
    dycp_size: int,
    num_cp_request: int,
    req_ids: Sequence[str],
    num_tokens_no_spec: Sequence[int],
    temperatures: Sequence[float | None],
    existing_generator_indices: Collection[int],
    model_seed: int | None,
    phase: str,
) -> Iterator[DyCPSampleSeed]:
    if dycp_size <= 1:
        return

    num_rows = min(
        int(num_cp_request or 0),
        len(req_ids),
        len(num_tokens_no_spec),
        len(temperatures),
    )
    for req_idx in range(num_rows):
        if req_idx in existing_generator_indices:
            continue

        temperature = temperatures[req_idx]
        if temperature is None or temperature <= _GREEDY_TEMPERATURE_EPS:
            continue

        yield DyCPSampleSeed(
            req_idx=req_idx,
            seed=make_dycp_sample_seed(
                model_seed,
                req_ids[req_idx],
                int(num_tokens_no_spec[req_idx]),
                phase,
            ),
        )


@contextmanager
def temporary_dycp_sample_generators(
    generators: MutableMapping[int, T],
    *,
    generator_factory: Callable[[int], T],
    dycp_size: int,
    num_cp_request: int,
    req_ids: Sequence[str],
    num_tokens_no_spec: Sequence[int],
    temperatures: Sequence[float | None],
    model_seed: int | None,
    phase: str,
) -> Iterator[None]:
    added_generator_indices: list[int] = []
    try:
        for sample_seed in iter_dycp_sample_seeds(
            dycp_size=dycp_size,
            num_cp_request=num_cp_request,
            req_ids=req_ids,
            num_tokens_no_spec=num_tokens_no_spec,
            temperatures=temperatures,
            existing_generator_indices=generators.keys(),
            model_seed=model_seed,
            phase=phase,
        ):
            generators[sample_seed.req_idx] = generator_factory(sample_seed.seed)
            added_generator_indices.append(sample_seed.req_idx)

        yield
    finally:
        for req_idx in added_generator_indices:
            generators.pop(req_idx, None)
