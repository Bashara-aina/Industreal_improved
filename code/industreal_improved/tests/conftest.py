import sys, os

# Make the project root a package so relative imports work
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

# Create package markers so relative imports resolve (project root + src sub-packages)
pkg_paths = [
    _root,
    os.path.join(_root, "src"),
    os.path.join(_root, "src", "training"),
    os.path.join(_root, "src", "models"),
    os.path.join(_root, "src", "data"),
    os.path.join(_root, "src", "evaluation"),
    os.path.join(_root, "src", "utils"),
]
for path in pkg_paths:
    init = os.path.join(path, "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()
