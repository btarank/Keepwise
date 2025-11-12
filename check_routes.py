from importlib import import_module
try:
    mod = import_module("app")
except Exception as e:
    print("ERROR importing app.py:", e)
    raise
app = getattr(mod, "app", None)
if not app:
    print("No Flask app object found in app.py")
else:
    print("Registered routes:")
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        print(rule.rule, sorted(rule.methods))
