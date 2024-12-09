import re
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def extract_text_from_log(log_file_path):
    pattern = r"[\d\.,;:'\"?!()_/-]+|blk|INFO|WARN|ERROR"
    text = []
    with open(log_file_path, 'r') as file:
        for line in file:
            text_line = re.sub(pattern, '', line).strip()
            text.append(text_line)
    return text

def extract_variable_from_log(log_file_path):
    Variables = []
    patterns = {
        'blockid': r'blk_-\d+|blk_\d+',                 
        'size':r'size\s+(\d+)',                         
        'packetresponse':r'PacketResponder\s+(\d+)',    
        'srcIP':r'from /(\b\d{1,3}(?:\.\d{1,3}){3}\b)|src: /(\b\d{1,3}(?:\.\d{1,3}){3}\b)|ask (\b\d{1,3}(?:\.\d{1,3}){3}\b)|(\d{1,3}(?:\.\d{1,3}){3}):?\d*\s+Served block|(\d{1,3}(?:\.\d{1,3}){3}):?\d*:Got exception',
        'srcPort':r'src: [^:]+:(\d+)|ask [^:]+:(\d+)|[^:]+:(\d+) Served block|[^:]+:(\d+):Got exception',            
        'dstIP':r'updated:\s+(\d{1,3}(?:\.\d{1,3}){3})|dest: /(\b\d{1,3}(?:\.\d{1,3}){3}\b)|to /(\b\d{1,3}(?:\.\d{1,3}){3}\b)|datanode(s) (\b\d{1,3}(?:\.\d{1,3}){3}\b)',
        'dstPort':r'updated: [^:]+:(\d+)|dest: [^:]+:(\d+)|datanode(s) [^:]+:(\d+)',              
        'address':r'NameSystem\.allocateBlock:\s+([^\.]+)',
    }
    with open(log_file_path, 'r') as file:
        for index, line in enumerate(file):
            for type, pattern in patterns.items():
                matches = re.findall(pattern, line)
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0] or match[1] or match[2] or match[3] or match[4] or match[5]
                        if not match:
                            continue
                    Variables.append((match, type, index))  
    return Variables

def save_to_txt(data,path):
    data.sort(key=lambda x: x[2]) 
    with open(path, 'w', encoding='utf-8') as file:
        for text, tag, line_number in data:
            file.write(f'{line_number}: {text} ({tag})\n')

            
if __name__ == '__main__':
    log_file_path = 'code/log/Data/test.log'
    Semantic_text = extract_text_from_log(log_file_path)
    Variables = extract_variable_from_log(log_file_path)
    # print(Variables)
    save_to_txt(Variables,'code/log/Data/res.txt')
