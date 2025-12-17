def widget_get(name, default):
    try:
        if value := dbutils.widgets.get(name):
            return value
    except Exception:
        pass
    dbutils.widgets.text(name, default)
    return default
