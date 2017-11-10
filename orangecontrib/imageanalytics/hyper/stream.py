# -*- coding: utf-8 -*-
"""
Temporary fix for hyper library until next version from 0.7.0 is out
"""
import logging

log = logging.getLogger(__name__)

# Define the largest chunk of data we'll send in one go. Realistically, we
# should take the MSS into account but that's pretty dull, so let's just say
# 1kB and call it a day.
MAX_CHUNK = 1024


class Stream(object):

    def send_data(self, data, final):
        """
        Send some data on the stream. If this is the end of the data to be
        sent, the ``final`` flag _must_ be set to True. If no data is to be
        sent, set ``data`` to ``None``.
        """
        # Define a utility iterator for file objects.
        def file_iterator(fobj):
            while True:
                data = fobj.read(MAX_CHUNK)
                yield data
                if len(data) < MAX_CHUNK:
                    break

        # Build the appropriate iterator for the data, in chunks of CHUNK_SIZE.
        if hasattr(data, 'read'):
            chunks = file_iterator(data)
        else:
            chunks = (data[i:i+MAX_CHUNK]
                      for i in range(0, len(data), MAX_CHUNK))

        # since we need to know when we have a last package we need to know
        # if there is another package in advance
        cur_chunk = None
        try:
            cur_chunk = next(chunks)
            while True:
                next_chunk = next(chunks)
                self._send_chunk(cur_chunk, False)
                cur_chunk = next_chunk
        except StopIteration:
            if cur_chunk is not None:  # cur_chunk none when no chunks to send
                self._send_chunk(cur_chunk, final)


    def _send_chunk(self, data, final):
        """
        Implements most of the sending logic.
        Takes a single chunk of size at most MAX_CHUNK, wraps it in a frame and
        sends it. Optionally sets the END_STREAM flag if this is the last chunk
        (determined by being of size less than MAX_CHUNK) and no more data is
        to be sent.
        """
        # If we don't fit in the connection window, try popping frames off the
        # connection in hope that one might be a window update frame.
        while len(data) > self._out_flow_control_window:
            self._recv_cb()

        # Send the frame and decrement the flow control window.
        with self._conn as conn:
            conn.send_data(
                stream_id=self.stream_id, data=data, end_stream=final
            )
        self._send_outstanding_data()

        if final:
            self.local_closed = True