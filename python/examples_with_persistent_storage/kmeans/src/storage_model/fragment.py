try:
    # dataClay
    from storage.api import StorageObject
except:
    try:
        # Hecuba
        from hecuba.storageobj import StorageObj as StorageObject
    except:
        # Redis
        from storage.storage_object import StorageObject


class Fragment(StorageObject):
    """
    @ClassField mat numpy.ndarray
    """
    pass
