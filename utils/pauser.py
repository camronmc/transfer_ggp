import code
import signal

paused = False


def set_pauser(locals, quit_callback=None):
    def handler(signal, frame):
        global paused
        if paused:
            return
        paused = True
        print('Ctrl-c received')
        q = input('Enter q to quit')
        if q.lower() == 'q':
            if quit_callback:
                quit_callback()
            exit(0)
        locs = dict(locals)
        code.interact(
            banner='Modify variables and exit',
            local=locs
        )
        del locs
        print('Continuing...')
        paused = False

    signal.signal(signal.SIGINT, handler)
