class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    @staticmethod
    def to_dict(obj):
        return obj.__dict__
