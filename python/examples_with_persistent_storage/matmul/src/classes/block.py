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


class Block(StorageObject):
    """
    @ClassField block numpy.ndarray
    """
