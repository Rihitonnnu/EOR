import winsound


def high():
    winsound.Beep(600, 200)


def low():
    winsound.Beep(300, 200)


def error():
    winsound.Beep(1000, 200)
    winsound.Beep(1000, 200)
