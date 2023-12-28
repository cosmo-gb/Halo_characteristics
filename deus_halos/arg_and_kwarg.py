from __future__ import annotations


class Example:

    def __init__(self, *args, **kwargs) -> None:
        for arg, param in kwargs.items():
            setattr(self, arg, param)
        # delattr, hasattr


if __name__ == '__main__':
    lol = Example(guillaume_iq=200)
    print(lol.guillaume_iq)
