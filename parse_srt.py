""" Parses .srt file and outputs only the text

Usage:
    parse_srt.py <FILE>
"""
import re
from bs4 import BeautifulSoup
from docopt import docopt

args = docopt(__doc__)


fh = open(args['<FILE>'], 'rb')
data = fh.read()
fh.close()

splits = [s.strip() for s in re.split(r'\n\s*\n', data) if s.strip()]
regex = re.compile(r'''(?P<index>\d+).*?(?P<start>\d{2}:\d{2}:\d{2},\d{3}) --> (?P<end>\d{2}:\d{2}:\d{2},\d{3})\s*.*?\s*(?P<text>.*)''', re.DOTALL)
for s in splits:
    r = regex.search(s)
    sub = re.sub(r"\n+", " ", r.group(4))
    text = BeautifulSoup(sub, 'html.parser')
    print text.get_text()
    
