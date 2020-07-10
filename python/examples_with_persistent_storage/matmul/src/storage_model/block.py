try:
    # dataClay and Redis
    from storage.api import StorageObject
except:
    # Hecuba
    from hecuba.storageobj import StorageObj as StorageObject


class Block(StorageObject):
    """
    @ClassField block numpy.ndarray
    """
    pass
