import os
import logging
from colorlog import ColoredFormatter

log = None
def setup_log():
    global log
    if log is None:
        formatter = ColoredFormatter(
        	"%(log_color)s%(name)s: %(message)s",
        	datefmt=None,
        	reset=True,
        	log_colors={
        		'DEBUG':    'cyan',
        		'INFO':     'green',
        		'WARNING':  'yellow',
        		'ERROR':    'red',
        		'CRITICAL': 'red,bg_white',
        	},
        	secondary_log_colors={},
        	style='%'
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        log = logging.getLogger('tycho')
        log.setLevel(os.environ.get('TYCHO_LOG_LEVEL', 'DEBUG'))
        log.addHandler(handler)

setup_log()
