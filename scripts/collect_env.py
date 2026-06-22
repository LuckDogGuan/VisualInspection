from __future__ import annotations

import importlib
import platform
import sys


def module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"NOT_INSTALLED {type(exc).__name__}: {exc}"
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    print(f"python={sys.version.replace(chr(10), ' ')}")
    print(f"executable={sys.executable}")
    print(f"platform={platform.platform()}")
    for name in ["torch", "torchvision", "PIL", "numpy", "cv2", "ultralytics"]:
        print(f"{name}={module_version(name)}")

    try:
        import torch

        print(f"torch_cuda={torch.version.cuda}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"cudnn={torch.backends.cudnn.version()}")
        print(f"gpu_count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                print(f"gpu{index}={torch.cuda.get_device_name(index)}")
    except Exception as exc:
        print(f"torch_details_error={type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
