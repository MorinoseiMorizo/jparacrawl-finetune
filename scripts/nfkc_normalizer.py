import sys
import unicodedata

for line in sys.stdin:
    print(unicodedata.normalize("NFKC", line.rstrip()))
