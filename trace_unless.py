import pdb


def trace_unless(condition, message, **kwds):
    'like assert condition, message; but enters debugger if condition fails'
    if condition:
        return
    print '+++++++++++++++'
    for k, v in kwds.iteritems():
        print k, v
    print message
    print '+++++++++++++++'
    pdb.set_trace()
