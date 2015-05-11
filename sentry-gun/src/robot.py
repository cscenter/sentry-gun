import time

__author__ = 'vasdommes'

import os
import subprocess


class Robot:
    def __init__(self, winscp_path, ip="192.168.1.1", login="root",
                 password=""):
        self.winscp_path = winscp_path
        self.ip = ip
        self.login = login
        self.password = password

    def open_and_trikRun(self, *commands):
        winscp = subprocess.Popen(self.winscp_path, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True, bufsize=0)
        open = 'open scp://{}:{}@{}\n'.format(self.login,
                                              self.password, self.ip)
        cd = winscp.stdin.write('cd trik\n')

        input = open + cd
        for command in commands:
            input += 'call trikRun -qws -s "{}" \n'.format(command)
        input += 'exit\n'
        return winscp.communicate(input)

    def open_winscp(self, winscp_path, ip="192.168.1.1", login="root",
                    password=""):
        self.winscp = subprocess.Popen(self.winscp_path, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       universal_newlines=True, bufsize=0)
        self.winscp.stdin.write(
            'open scp://{}:{}@{}\n'.format(self.login, self.password, self.ip))
        self.winscp.stdin.write('cd trik\n')

    def trikRun(self, command, winscp=None):
        if winscp is None:
            winscp = self.winscp
        winscp.stdin.write('call trikRun -qws -s "{}" \n'.format(command))


    def close_winscp(self):
        if self.winscp is None:
            return
        self.winscp.stdin.close()
        self.winscp.terminate()


def robo_connect(winscp_path, ip="192.168.1.1", login="root",
                 password=""):
    winscp = subprocess.Popen(winscp_path, stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True, bufsize=0)
    # res = winscp.communicate('help\nexit\n')
    # print(res)
    winscp.stdin.write('option batch abort\n')

    winscp.stdin.write('help\n')
    # winscp.stdin.flush()
    # winscp.stdin.close()
    print('help sent...')
    lines = winscp.stdout.readlines()
    print('out1.................\n{}'.format(''.join(lines)))

    winscp.stdin.write('session\n')
    # winscp.stdin.flush()
    # winscp.stdin.close()
    lines = winscp.stdout.readlines()
    print('out1.................\n{}'.format(''.join(lines)))


    # out, err = winscp.communicate('help')
    # # input='open scp://{}:{}@{}'.format(login, password, ip))
    # print(out)
    # print('-----------------')
    # print(err)
    # winscp.communicate('exit')


if __name__ == '__main__':
    # robo_connect('cmd')
    # robo_connect('C:\TRIKStudio\winscp\WinSCP.com')
    robo_connect(os.path.join(os.pardir, 'putty', 'plink.exe'))