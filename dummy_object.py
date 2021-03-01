import uuid


class DummyObj:
    def __init__(self, classId=None):
        self.UUID = uuid.uuid1()
        self.classId = classId
