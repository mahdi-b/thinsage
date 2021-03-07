import uuid


class DummyObj:
    def __init__(self, class_id=None, index=-1):
        self.UUID = uuid.uuid1()
        self.class_id = class_id
        self.original_index = index
