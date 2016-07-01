import atexit
import time


class Timer(object):
    def __init__(self):
        self._program_start_clock = time.clock()  # processor time in seconds
        self._program_start_time = time.time()  # time in seconds since the epoch
        self._program = (self._program_start_clock, self._program_start_time)
        self._lap = (self._program_start_clock, self._program_start_time)
        atexit.register(self.endlaps)

    # initial API
    def elapsed_cpu_seconds(self):
        return time.clock() - self._program_start_clock

    def elapsed_wallclock_seconds(self):
        return time.time() - self._program_start_time

    # second API (keep first for backwards compatibility)
    def clock_time(self):
        return (time.clock(), time.time())

    def lap(self, s):
        'print time of current lap'
        # inspired by Paul McGuire's timing.py
        # ref: http://stackoverflow.com/questions/1557571/how-to-get-time-of-a-python-program-execution

        def toStr(t):
            'convert seconds to hh:mm:ss.sss'
            # this code from Paul McGuire!
            return '%d:%02d:%02d.%03d' % reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                                                [(t * 1000,), 1000, 60, 60])

        def diff(start, now):
            return (
                toStr(now[0] - start[0]),
                toStr(now[1] - start[1])
            )

        clue = '=' * 50
        clock_time = self.clock_time()
        print clue
        print 'lap: %s' % s
        print 'cumulative %s cpu %s wallclock' % diff(self._program, clock_time)
        print 'lap        %s cpu %s wallclock' % diff(self._lap, clock_time)
        print clue
        print
        self._lap = clock_time  # reset lap timer

    def endlaps(self):
        self.lap('**End Program**')
