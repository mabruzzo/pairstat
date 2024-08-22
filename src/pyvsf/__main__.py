# this is a short script used to let you query the backend that the package was
# compiled with

from ._kernels_cy import compiled_with_openmp


def main():
    has_openmp = compiled_with_openmp()
    if has_openmp:
        backend = "openmp-cpu"
    else:
        backend = "serial"
    print(f"backend: {backend}")


if __name__ == "__main__":
    main()
