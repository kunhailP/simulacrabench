import html,re,sys
s=open(sys.argv[1],encoding='utf-8',errors='replace').read()
s=re.sub(r'(?is)<(script|style|svg)[^>]*>.*?</\1>','',s)
s=re.sub(r'(?i)<br\s*/?>|</(p|div|h\d|li|tr|section)>','\n',s)
s=re.sub(r'<[^>]+>',' ',s)
s=html.unescape(s)
s=re.sub(r'[ \t]+',' ',s); s=re.sub(r'\n\s*\n+','\n\n',s)
print(s)
