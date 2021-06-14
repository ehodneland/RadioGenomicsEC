

class Struct(dict):
    # take a set of prop=value assignments as arguments
    def __init__(self, **kwargs):
        # initialize the dictionary part with these assignments
        super().__init__(**kwargs)
        # use the property values as the class dictionary, i.e.
        # turn the items in the dictionary into fields of the
        # object!
        self.__dict__ = self
