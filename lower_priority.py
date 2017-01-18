'lower priority of current process'
import os
import win32api
import win32process
import win32con


def lower_priority():
    # ref: http://stackoverflow.com/questions/1023038/change-process-priority-in-python-cross-platform
    assert os.name in ('nt', 'posix'), os.name
    if os.name == 'nt':
        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
    elif os.name == 'posix':
        os.nice(1)

if __name__ == '__main__':
    lower_priority()