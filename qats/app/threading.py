import sys
import traceback
from qtpy.QtCore import QRunnable, Signal as QSignal, QObject, Slot as QSlot


class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        `tuple` (exctype, value, traceback.format_exc() )

    result
        `object` data returned from processing, anything
    """
    finished = QSignal()
    error = QSignal(tuple)
    result = QSignal(object)


class Worker(QRunnable):
    """
    Worker thread inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    Parameters
    ----------
    callback: function
        The function callback to run on this worker thread. Supplied args and kwargs will be passed through to the runner.
    *args
        Variable length argument list passed to callback function.
    **kwargs
        Arbitrary keyword arguments passed to callback function.
    """
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @QSlot()
    def run(self):
        """
        Initialise the runner function with passed args, kwargs.
        """
        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            # catch exception and return traceback
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            # Return the result of the processing
            self.signals.result.emit(result)
        finally:
            # Done
            self.signals.finished.emit()
