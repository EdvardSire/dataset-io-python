import os, sys, gc, struct, threading
from pathlib import Path
from hashlib import sha256
from multiprocessing import resource_tracker as _mprt
from multiprocessing import shared_memory as _mpshm

import numpy as np
import cv2 


b2gb = 1024**3


class ShmDatasetConsumer():
    def __init__(self, image_dir: Path):
        shm_name = sha256(image_dir.home().__str__().encode()).hexdigest()
        self._shm = SharedMemory(name=shm_name, create=False, track=False)
        ms, n, h, w, c = struct.unpack_from('5i', self._shm.buf, offset=0)
        self.data =  np.frombuffer(self._shm.buf, dtype=np.uint8, offset=ms).reshape((n, h, w, c))

    def cleanup(self):
        del self.data
        self._shm.close()


class ShmDataset():
    @staticmethod
    def load_from_path(image_dir: Path):
        sd = ShmDataset()
        shm_name = sha256(image_dir.home().__str__().encode()).hexdigest()
        image_paths = sorted(image_dir.iterdir())
        n, h, w, c, datatype = sd.profile_dataset(image_paths)
        slab_size_bytes = n*h*w*c*np.dtype(datatype).itemsize + sd.metadata_size()
        sd.assert_feasible_allocation(slab_size_bytes)
        shm = SharedMemory(create=True, size=slab_size_bytes, name=shm_name)
        struct.pack_into('i', shm.buf, 0, sd.metadata_size())
        shm.buf[4:4+(4*4)] = struct.pack('4i', n, h, w, c)
        gc.collect()

        arr = np.ndarray((n, h, w, c), dtype=datatype, buffer=shm.buf, offset=sd.metadata_size())
        for i, image_path in enumerate(image_paths):
            img = cv2.imread(image_path.__str__())

            if img.shape != (h, w, c):
                raise ValueError(f"Image {image_path} has unexpected shape {img.shape}")
            arr[i] = img  

        try:
            print(f"Spinning with allocated slab: {shm_name} ; size {slab_size_bytes/b2gb:.2f} GB")
            while True:
                pass
        except KeyboardInterrupt:
            shm.close()
            shm.unlink()

    @staticmethod
    def metadata_size():
        # (int32) 4 bytes metadata_size_bytes 
        # (int32) 4 bytes n_images
        # (int32) 4 bytes h_image
        # (int32) 4 bytes w_image
        # (int32) 4 bytes c_image

        # 20 bytes + 44 bytes padding = 64 bytes
        return 64

    @staticmethod
    def shm_limit_and_used_bytes(path="/dev/shm"):
        # findmnt -o AVAIL,USED /dev/shm
        # sudo mount -o remount,size=XXG /dev/shm
        st = os.statvfs(path)
        limit = st.f_bavail * st.f_frsize  # available bytes to non-root
        used  = (st.f_blocks - st.f_bfree) * st.f_frsize  # used bytes
        return limit, used

    @staticmethod
    def available_system_memory_bytes(meminfo = Path("/proc/meminfo")) -> float:
        with open(meminfo) as f:
            meminfo = {line.split(':')[0]: int(line.split()[1]) for line in f}
            return meminfo["MemAvailable"] * 1024

    @staticmethod
    def profile_dataset(image_paths: list[Path]):
        i1 = cv2.imread(image_paths[0].__str__())
        h, w, c = i1.shape
        datatype = i1.dtype
        n = len(image_paths)

        return n, h, w, c, datatype

    @staticmethod
    def assert_feasible_allocation(needed_bytes):
        sd = ShmDataset()
        avail_system_bytes = sd.available_system_memory_bytes()
        shm_limit_bytes, _ = sd.shm_limit_and_used_bytes()
        pressure = needed_bytes/avail_system_bytes

        if needed_bytes > shm_limit_bytes:
            raise RuntimeError(f"Increase /dev/shm size, likely: `sudo mount -o remount,size={1.2*needed_bytes/b2gb:.0f}G /dev/shm`")

        if pressure > 0.9:
            raise RuntimeError(f"""Not comfortable allocating:
                                       pressure     {pressure:.2f}
                                       needed/avail/shm-limit {needed_bytes/b2gb:.2f}/{avail_system_bytes/b2gb:.2f}/{shm_limit_bytes/b2gb:.2f} GB""")
        print(f"Memory pressure: {pressure:.2f}")
        return


# https://github.com/python/cpython/issues/82300#issuecomment-2169035092
if sys.version_info >= (3, 13):
    SharedMemory = _mpshm.SharedMemory
else:
    class SharedMemory(_mpshm.SharedMemory):
        __lock = threading.Lock()

        def __init__(
            self, name: str | None = None, create: bool = False,
            size: int = 0, *, track: bool = True
        ) -> None:
            self._track = track

            # if tracking, normal init will suffice
            if track:
                return super().__init__(name=name, create=create, size=size)

            # lock so that other threads don't attempt to use the
            # register function during this time
            with self.__lock:
                # temporarily disable registration during initialization
                orig_register = _mprt.register
                _mprt.register = self.__tmp_register

                # initialize; ensure original register function is
                # re-instated
                try:
                    super().__init__(name=name, create=create, size=size)
                finally:
                    _mprt.register = orig_register

        @staticmethod
        def __tmp_register(*args, **kwargs) -> None:
            return

        def unlink(self) -> None:
            if _mpshm._USE_POSIX and self._name:
                _mpshm._posixshmem.shm_unlink(self._name)
                if self._track:
                    _mprt.unregister(self._name, "shared_memory")
