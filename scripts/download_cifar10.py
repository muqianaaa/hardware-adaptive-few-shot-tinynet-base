from pathlib import Path

from torchvision.datasets import CIFAR10


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "data" / "raw" / "cifar10"
    target.mkdir(parents=True, exist_ok=True)

    for train in (True, False):
        split = "train" if train else "test"
        print(f"Downloading CIFAR-10 {split} split to {target} ...")
        CIFAR10(root=str(target), train=train, download=True)

    print(f"CIFAR-10 is ready under: {target}")


if __name__ == "__main__":
    main()
