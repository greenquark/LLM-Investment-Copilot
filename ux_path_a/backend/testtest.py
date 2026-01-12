import main
print("main file:", main.__file__)
print("app:", main.app)
paths = []
for r in main.app.routes:
    p = getattr(r, "path", None)
    m = getattr(r, "methods", None)
    if p:
        paths.append((p, m))
for p,m in sorted(paths):
    print(p, m)
