import socket
import sys
import re
import time
from propnet.propnet import make_propnet

def run_player(player_class, name='player'):
    p = player_class(name, int(sys.argv[1]))
    p.run()

class Player:
    CONTENT_LENGTH_RE = re.compile('\r\ncontent-length: *(\\d+)', re.IGNORECASE)
    START_RE = re.compile(r'\(start +(\w+) +(\w+) +(.+) +(\d+) +(\d+)\)')
    PLAY_RE = re.compile(r'\(play +(\w+) +(.+)\)')
    STOP_RE = re.compile(r'\(stop +(\w+) +(.+)\)')
    ABORT_RE = re.compile(r'\(abort +(\w+)\)')
    def __init__(self, name, PORT):
        self.PORT = PORT
        self.name = name
        self.propnet = None

    def run(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host = 'localhost'#socket.gethostname()
        self.socket.bind((host, self.PORT))
        self.socket.listen(5)
        print('Player %s started listening at %s:%d' % (self.name, host, self.PORT))
        stop = False
        while not stop:
            c, addr = self.socket.accept()
            msg = ''
            header_length = -1
            while header_length == -1:
                buf = c.recv(4096).decode('utf-8')
                msg += buf
                header_length = msg.find('\r\n\r\n')
            header_length += 4

            content_length = self.CONTENT_LENGTH_RE.search(msg)
            if content_length is None:
                raise ValueError('No content length string in input: ' + msg)
            content_length = int(content_length.group(1))

            msg_length = header_length+content_length

            while len(msg) < msg_length:
                buf = c.recv(4096).decode('utf-8')
                msg += buf

            result, stop = self.handle_request(msg[header_length:])
            result = result.encode('utf-8')
            response = (
                b'HTTP/1.0 200 OK\r\n' +
                b'Content-type: text/acl\r\n' +
                (b'Content-length: %d' % len(result)) +
                b'\r\n\r\n' +
                result
            )
            c.send(response)
            c.close()


    def handle_request(self, msg):
        msg_type, args = self.parse_msg(msg.lower())
        if msg_type == 'info':
            self.info()
            return 'ready', False
        elif msg_type == 'start':
            self._start(*args)
            return 'ready', False
        elif msg_type == 'play':
            move = self._play(*args)
            return move, False
        elif msg_type == 'stop':
            self._stop(*args)
            return 'done', True
        elif msg_type == 'abort':
            self.abort(*args)
            return 'done', True
        else:
            raise ValueError('Unrecognised input: ' + msg)

    def parse_msg(self, msg):
        if msg == 'info()':
            return 'info', []
        start = self.START_RE.search(msg)
        if start:
            return 'start', start.groups()
        play = self.PLAY_RE.search(msg)
        if play:
            return 'play', play.groups()
        stop = self.STOP_RE.search(msg)
        if stop:
            return 'stop', stop.groups()
        abort = self.ABORT_RE.search(msg)
        if abort:
            return 'abort', abort.groups()
        return 'unknown', []

    def info(self):
        print('info!')
        pass

    def _start(self, id, role, rules, startclock, playclock):
        start = time.time()
        self.match_id = id
        self.role = role
        self.rules = rules
        self.propnet = make_propnet(rules[1:-1], id+role)
        self.startclock = int(startclock)
        self.playclock = int(playclock)
        print('Game', id, 'started!')
        print('I am', role)
        print('Startclock:', startclock)
        print('Playclock:', playclock)
        self.start(start + self.startclock)

    def start(self, endtime):
        pass

    def _play(self, id, move):
        start = time.time()
        if move != 'nil':
            self.propnet.do_actions(move)
            self.propnet.do_step()
        my_move = self.play(start + self.playclock).strip()
        return my_move

    def play(self):
        raise NotImplementedError()

    def _stop(self, id, move):
        self.propnet.do_actions(move)
        self.propnet.do_step()
        print('Game finished, results (I am %s):' % self.role)
        print(*(x
                for x in self.propnet.goal
                if x.eval(self.propnet.data)), sep='\n')

    def stop(self):
        pass

    def abort(self, id):
        pass

