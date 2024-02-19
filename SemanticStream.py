import queue

from transformers.generation.streamers import BaseStreamer


class SemanticStreamer(BaseStreamer):
    def __init__(self, chunk_size) -> None:
        self.output_ids = []
        self.chunk_size = chunk_size
        self._queue = queue.Queue()
        self._finished = False

    def put(self, item) -> None:
        if self._finished:
            return
        if len(self.output_ids) == self.chunk_size:
            self.output_ids.append(item)
            self._queue.put_nowait(self.output_ids)
            self.output_ids = []
        else:
            self.output_ids.append(item)

    def end(self) -> None:
        if len(self.output_ids) > 0:
            self._queue.put_nowait(self.output_ids)
        self._queue.put_nowait(StopIteration)
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __iter__(self):
        return self

    def __next__(self):
        result = self._queue.get()
        if result is StopIteration:
            raise StopIteration
        elif isinstance(result, Exception):
            raise result
        return result
