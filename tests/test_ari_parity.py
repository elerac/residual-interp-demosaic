import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demosaic import mosaic_bayer
from demosaic.algorithms import (
    demosaic_ari,
    demosaic_ari2,
    demosaic_mlri,
    demosaic_mlri2,
    demosaic_ri,
)
from demosaic.algorithms.ari import _cubic_kernel


PATTERNS = ("rggb", "grbg", "gbrg", "bggr")
ALGORITHMS = {
    "ARI": demosaic_ari,
    "ARI2": demosaic_ari2,
}
RI_FAMILY_ALGORITHMS = {
    "RI": demosaic_ri,
    "MLRI": demosaic_mlri,
    "MLRI2": demosaic_mlri2,
}
ALL_ALGORITHMS = {
    **ALGORITHMS,
    **RI_FAMILY_ALGORITHMS,
}
EXPECTED_DIGESTS = {
    ("ARI", "rggb"): "405702b60b56101f0ff746bd53b0f2eb6ca2c5bcd822ef5ed43b675339065705",
    ("ARI", "grbg"): "5f64913aec4fa27d62dee04eff40b8fdb2bb6eac2a2fdaf96b6bbf3bf5f8104f",
    ("ARI", "gbrg"): "38a8c73593bd2dcea68b294cec20e53e54c820a16179420bfec5052b8dfe600b",
    ("ARI", "bggr"): "22481f60addcef2e14194fca15fc9a6aea7a70a0908e3ebed748ac0fee9c8eab",
    ("ARI2", "rggb"): "f331e419e3bec07ffe585f47a588c6acb3503e1efdd9578f2bbecf3f02d1cb95",
    ("ARI2", "grbg"): "c9e36ae4df5f0d8a51589d08540d31a80ed2ee53abd1830fd26820abf62d24dc",
    ("ARI2", "gbrg"): "8b0868124b0a1a01ee26f9398c7a871ede70d755be141bdd42df14f9b031c827",
    ("ARI2", "bggr"): "d0842b92366bf2f20310ad55cbfb2ff8db4400b13b77001384ab958d58422eee",
}
EXPECTED_GRADIENT_DIGESTS = {
    ("ARI", "rggb"): "c66002267967d2657ff9ff09517637f2d724b1be3fe709f659220cfa62f0c82e",
    ("ARI", "grbg"): "2e942a7cea7214bf249b592244cceecc07aad258b0fa5d9fe3cda8834fee47ba",
    ("ARI", "gbrg"): "aeb7bee091b0defcc302ed462a31ac851cda3724f2b7b6bf4f5a1a1546db8650",
    ("ARI", "bggr"): "6505c70d2238cf5cc828aaf34be23a51e3251922bb94da6d3a4aaf53b1d0a8a5",
    ("ARI2", "rggb"): "497776c19aa48d07dd9480554b9ad2c1358135bd6886040711990037d188d8b5",
    ("ARI2", "grbg"): "190180ee2b301e4acec0bb265c00459489d9d29bde425ba15a999d0887e71b0c",
    ("ARI2", "gbrg"): "38d2e0d77eb535ad828e10aeabf1e45186383fe2740e3c466be363ab6cfa158b",
    ("ARI2", "bggr"): "89873385aea76c751691a1961848d86908bc13806731ac5073c5b035a30766c1",
}
RI_FAMILY_EXPECTED_DIGESTS = {
    ("RI", "rggb"): "e02f9df2e66b761ff8df407c88805ddf32637e37315ba8b68d2cd356053ac5bd",
    ("RI", "grbg"): "301122b2c48849cc390f621b7a9a919f4de6a944a5c4f420053f3c7c2788dd5f",
    ("RI", "gbrg"): "fe24b791a30cfc52cd5fe9c1840c7b636f7f9ca32668189000c46a1bd7e09809",
    ("RI", "bggr"): "9dda235882194658ef7971acaf5e86598a2fcc4daf102d8f03c78a309694ac66",
    ("MLRI", "rggb"): "3cb6e654959598e4107308ed656df471ed6e70739482a1c6330244369b0fc207",
    ("MLRI", "grbg"): "31b7ceab5fcc77086cb8dfe0515336f2993f3e541e1348a4f0bb4cb976361a44",
    ("MLRI", "gbrg"): "d8fc59ea2200bc434e516beae86b9f0d89ea469a7892dfd73a094d22545062ad",
    ("MLRI", "bggr"): "1d36fecbf04a7a44a4c39f54021b6a06d61066d5843687b96c92e518f73d645d",
    ("MLRI2", "rggb"): "5579c8d364434fef49a42852b9a4d39dad2af1c1217a18380e6e5777cb5c7064",
    ("MLRI2", "grbg"): "bfc4ab8b47d6babf33489fde9d2ab55812c5c6b37a36b9621fd11efff6fc3e03",
    ("MLRI2", "gbrg"): "c9cc9be4f2fa1033c5c080c2b8ba19431eebca32b95e0b6e1c02e0720d014a0b",
    ("MLRI2", "bggr"): "9184bec8bea1ae7e9cb24b46e4104b4e895f7d1f5ae1a74c4a4ca5d25fde5a11",
}
RI_FAMILY_EXPECTED_GRADIENT_DIGESTS = {
    ("RI", "rggb"): "b8e0a3ad9aa67c06f6c4870ab0e5e7af961e6bacb723ee2270e7e7513cf97eb7",
    ("RI", "grbg"): "95290a2f62d6b070879937d8bf09e6b47bafe88a7d77d2821cebb2d2e0fb2cd4",
    ("RI", "gbrg"): "e719abe9aa90dfeb84e4c646361a8c6dba533199dc4426cd991aa7d2730816ad",
    ("RI", "bggr"): "872db918777b02d3742a32a79a3fc9e1c3d4ed4f9875f98667e177539c0485fe",
    ("MLRI", "rggb"): "d0c876ab91687edfac41248cd043e551a317899eb2648213d13e2449f13afc5a",
    ("MLRI", "grbg"): "4d0f815a3a2b030db6f38cda1d03c8050ac59316f2a48621701a365371e8ae22",
    ("MLRI", "gbrg"): "a0c01a636a578292c0c569fa674f52db3ec7351597a368d8bbe31e4cb2cb7697",
    ("MLRI", "bggr"): "b006a59bfc7ecaa17c87553d98bfb70e99dd9b0b692f7ae839100289f8c22722",
    ("MLRI2", "rggb"): "4ac7f4e04d7088e8f23b7928460e7787a91a6e54673dd90b057cb37907585b4d",
    ("MLRI2", "grbg"): "64ad21671ede32b1bff6b170e7e8e1c2ebead6738d62fc353401ce911127b8cd",
    ("MLRI2", "gbrg"): "feed33f1c3c47c8259ce8f102cd8df93153bc91bd0e5e4db402ee94b72e98087",
    ("MLRI2", "bggr"): "c4668623f42bb50532a4a7d5e1d5cf5982424fe85f7075d644d0611bc3399db0",
}
EXPECTED_CHECKER_DIGESTS = {
    ("ARI", "rggb"): "dca2942a02c9e33b8b142ac29db534d28390bc16cb1ff6f2ebb1259c8e180689",
    ("ARI", "grbg"): "b05c1819e8932e02d10609b4e74efe88863a197ea32d175c2024a80682168bc0",
    ("ARI", "gbrg"): "1acc2ee64aec7e8c7699e62b25559b4d4c58183f62d41986c29b96b8bb5928ea",
    ("ARI", "bggr"): "b678ad9e52cda080f89cf78be79d212f1c304c698ca6dad29a13d37d3d6afeee",
    ("ARI2", "rggb"): "57889ce852023d133eec2b3687ef4883be0ae04eb4dae6781587299ce6f33204",
    ("ARI2", "grbg"): "04f95208dc5236be2ab83d8651391820d77e73c659983cfaa996407877aa9e52",
    ("ARI2", "gbrg"): "9d9ae03e6d12a896e98e4480ba895a9d05b14d75e69cc75cc95ae81576488eb9",
    ("ARI2", "bggr"): "5aa62428b27b790458069882193c7bd7a209b64104e8bff8f81210789b505226",
    ("RI", "rggb"): "fb851195998e6aeaeddd85ca02df619f9b0f3fab6cb098d00b430863d7f17840",
    ("RI", "grbg"): "ed02f8a6fcbb08c2ce4c1dd57bec51cabfe4e8a027f2d7b5abe513db835074f6",
    ("RI", "gbrg"): "61c6b3e09fa50a8669909a9c13a976ab277709d1b2a76b2742299c857e5fdb17",
    ("RI", "bggr"): "17cae8df24f6a53685768c204154ecdc4c9463cacc07382d6a4d7981bd578b48",
    ("MLRI", "rggb"): "b1370295a705ac99e41d06daed788dd2df3607e82dadad40d66b79a2b4af70cb",
    ("MLRI", "grbg"): "3e8d67c09d3cd9c6823a86a53ce079f7d651283adf0bf7ef5f1629a3e156ce16",
    ("MLRI", "gbrg"): "1cf62e32cade4425f125b8aeaf3b3558ec0039f158874096fc5316f55383b7b8",
    ("MLRI", "bggr"): "0675d75b0a274845bc6b85ed4337843611212fff127e2cfcbab38733f6d7bad8",
    ("MLRI2", "rggb"): "ad75e05f50800bb8be436c49af159484193d27c016f084acfd7874962dec67ba",
    ("MLRI2", "grbg"): "2e49633d1b59e031d3064710602f1257db8b73887bd5362b74bac338e24e4116",
    ("MLRI2", "gbrg"): "94d651f6219ddce8ec7f2ac046b18641db31c51cb13ed3a921f312e78943ac0b",
    ("MLRI2", "bggr"): "6209e44b7cfff05f1f4eccac5fc0a421834d3af9e10f3284525b726fe10ab432",
}


def _deterministic_rgb(shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=np.int64)
    image = np.empty((*shape, 3), dtype=np.float64)
    image[:, :, 0] = (17 * rows + 31 * cols + (rows * cols) % 29) % 256
    image[:, :, 1] = (37 * rows + 11 * cols + np.bitwise_xor(rows, cols) * 3) % 256
    image[:, :, 2] = (13 * rows + 23 * cols + (rows - cols) ** 2) % 256
    return image


def _gradient_rgb(shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=np.float64)
    denom_r = max(shape[0] - 1, 1)
    denom_c = max(shape[1] - 1, 1)
    image = np.empty((*shape, 3), dtype=np.float64)
    image[:, :, 0] = cols * 255.0 / denom_c
    image[:, :, 1] = rows * 255.0 / denom_r
    image[:, :, 2] = (rows / denom_r + cols / denom_c) * 127.5
    return image


def _checker_rgb(shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=np.int64)
    image = np.empty((*shape, 3), dtype=np.float64)
    image[:, :, 0] = ((rows + cols) % 2) * 255.0
    image[:, :, 1] = ((rows // 2 + cols // 3) % 2) * 255.0
    image[:, :, 2] = ((rows // 3 + cols // 2) % 2) * 255.0
    return image


def _final_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _digest(image: np.ndarray) -> str:
    image = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(image.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(image.dtype.str.encode("ascii"))
    digest.update(b"|")
    digest.update(image.tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize(
    ("x", "expected"),
    ((0.0, 1.0), (0.5, 0.5625), (1.0, 0.0), (1.5, -0.0625), (2.0, 0.0), (2.5, 0.0)),
)
def test_cubic_kernel_is_even_and_compact(x: float, expected: float):
    assert _cubic_kernel(x) == pytest.approx(expected)
    assert _cubic_kernel(-x) == pytest.approx(expected)


@pytest.mark.parametrize("algorithm", ("ARI", "ARI2"))
@pytest.mark.parametrize("pattern", PATTERNS)
def test_ari_final_uint8_output_hash_stability(algorithm: str, pattern: str):
    rgb = _deterministic_rgb()
    mosaic, mask = mosaic_bayer(rgb, pattern)

    output = _final_uint8(ALGORITHMS[algorithm](mosaic, mask, pattern))

    assert output.shape == rgb.shape
    assert output.dtype == np.uint8
    assert _digest(output) == EXPECTED_DIGESTS[(algorithm, pattern)]


@pytest.mark.parametrize("algorithm", ("ARI", "ARI2"))
@pytest.mark.parametrize("pattern", PATTERNS)
def test_ari_gradient_uint8_output_hash_stability(algorithm: str, pattern: str):
    rgb = _gradient_rgb()
    mosaic, mask = mosaic_bayer(rgb, pattern)

    output = _final_uint8(ALGORITHMS[algorithm](mosaic, mask, pattern))

    assert output.shape == rgb.shape
    assert output.dtype == np.uint8
    assert _digest(output) == EXPECTED_GRADIENT_DIGESTS[(algorithm, pattern)]


@pytest.mark.parametrize("algorithm", ("RI", "MLRI", "MLRI2"))
@pytest.mark.parametrize("pattern", PATTERNS)
def test_ri_family_final_uint8_output_hash_stability(algorithm: str, pattern: str):
    rgb = _deterministic_rgb()
    mosaic, mask = mosaic_bayer(rgb, pattern)

    output = _final_uint8(RI_FAMILY_ALGORITHMS[algorithm](mosaic, mask, pattern))

    assert output.shape == rgb.shape
    assert output.dtype == np.uint8
    assert _digest(output) == RI_FAMILY_EXPECTED_DIGESTS[(algorithm, pattern)]


@pytest.mark.parametrize("algorithm", ("RI", "MLRI", "MLRI2"))
@pytest.mark.parametrize("pattern", PATTERNS)
def test_ri_family_gradient_uint8_output_hash_stability(algorithm: str, pattern: str):
    rgb = _gradient_rgb()
    mosaic, mask = mosaic_bayer(rgb, pattern)

    output = _final_uint8(RI_FAMILY_ALGORITHMS[algorithm](mosaic, mask, pattern))

    assert output.shape == rgb.shape
    assert output.dtype == np.uint8
    assert _digest(output) == RI_FAMILY_EXPECTED_GRADIENT_DIGESTS[(algorithm, pattern)]


@pytest.mark.parametrize("algorithm", ("ARI", "ARI2", "RI", "MLRI", "MLRI2"))
@pytest.mark.parametrize("pattern", PATTERNS)
def test_checker_uint8_output_hash_stability(algorithm: str, pattern: str):
    rgb = _checker_rgb()
    mosaic, mask = mosaic_bayer(rgb, pattern)

    output = _final_uint8(ALL_ALGORITHMS[algorithm](mosaic, mask, pattern))

    assert output.shape == rgb.shape
    assert output.dtype == np.uint8
    assert _digest(output) == EXPECTED_CHECKER_DIGESTS[(algorithm, pattern)]
