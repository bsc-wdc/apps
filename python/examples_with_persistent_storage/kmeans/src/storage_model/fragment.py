try:
    # dataClay and Redis
    from storage.api import StorageObject
except:
    # Hecuba
    from hecuba.storageobj import StorageObj as StorageObject


class Fragment(StorageObject):
    """
    @ClassField mat numpy.ndarray
    """
    pass
