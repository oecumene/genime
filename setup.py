from pathlib import Path
from setuptools import setup


def main():
    package_name = "genime"
    package_path = Path(__file__).parent.absolute()
    print(package_path)

    packages = {
        str(p.parent.relative_to(package_path)).replace('/', '.')
        for p in (package_path / package_name).rglob("__init__.py")
    }

    setuptools.setup(
        name=package_name,
        version="0.0.1",
        description="Generative Anime",
        packages=packages,
        package_dir={package_name: package_name},
        include_package_data=True,
        package_data={package_name: ["genime/data"]},
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
    )


if __name__ == '__main__':
    main()

