import time

__author__ = 'vasdommes'

import os
import subprocess

# M1 up +
# M2 shoot +
# M4 letf +
class Robot:
    def __init__(self, conn, angle_to_encoder):
        self.conn = conn = RobotConnector('C:\TRIKStudio\winscp\WinSCP.com')
        conn.open_winscp()
        time.sleep(20)
        self.angle_to_encoder = angle_to_encoder

    def rotate_and_shoot(self, alpha, phi):
        rotate_horiz = self._rotate_encoder_command('B4', 'M4', phi)
        rotate_vert = self._rotate_encoder_command('B1', 'M1', alpha)
        shoot = self._rotate_motor_command('M2', 1000)
        script = self._build_script(rotate_horiz, rotate_vert, shoot)
        self.conn.trikRun(script)

    def close(self):
        self.conn.close_winscp()

    def _build_script(self, *commands):
        start = 'var main = function() { '
        body = ' '.join(list(commands))
        end = ' return; }'
        return start + body + end

    def _rotate_encoder_command(self, encoder_name, motor_name, angle):
        encoder_angle = angle * self.angle_to_encoder
        enc = 'brick.encoder({})'.format(encoder_name)
        motor = 'brick.motor({})'.format(motor_name)
        init = 'var e = {}.read(); '.format(enc)
        if encoder_angle >= 0:
            on = motor + '.setPower(100);'
            cycle = 'while ({}.read() - e < {}) {  script.wait(1); }'.format(
                enc, encoder_angle)
        else:
            on = motor + '.setPower(-100);'
            cycle = 'while ({}.read() - e > {}) {  script.wait(1); }'.format(
                enc, encoder_angle)
        off = motor + '.powerOff();'

        return ' '.join([init, on, cycle, off])

    def _rotate_motor_command(self, motor_name, time_ms, power=100):
        motor = 'brick.motor({})'.format(motor_name)
        on = motor + '.setPower({});'.format(power)
        wait = 'script.wait({});'.format(time_ms)
        off = motor + '.powerOff();'
        return ' '.join([on, wait, off])


class RobotConnector:
    def __init__(self, winscp_path, ip='192.168.1.1', login='root',
                 password=""):
        """

        :type winscp_path: str | Unicode
        """
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
        cd = 'cd trik\n'

        input = open + cd
        for command in commands:
            input += 'call trikRun -qws -s "{}" \n'.format(command)
        input += 'exit\n'
        return winscp.communicate(input)

    def open_winscp(self):
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
