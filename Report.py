'Report writer'

import pdb


class Report(object):
    def __init__(self):
        self.lines = []

    def append(self, line):
        self.lines.append(line)
        print line

    def extend(self, lines):
        for line in lines:
            self.append(line)

    def write(self, path):
        f = open(path, 'w')
        for line in self.lines:
            try:
                f.write(str(line))
            except:
                print line
                print type(line)
                pdb.set_trace()
            f.write('\n')
        f.close()

    def lines(self):
        return self.lines
