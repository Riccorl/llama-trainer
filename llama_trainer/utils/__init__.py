import importlib


def is_package_available(package_name: str) -> bool:
    """
    Check if a package is available.

    Args:
        package_name (`str`): The name of the package to check.
    """
    return importlib.util.find_spec(package_name) is not None
