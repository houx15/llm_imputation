def log(log_file, text):
    with open(log_file, "a", encoding="utf8") as wfile:
        wfile.write(text + "\n")
